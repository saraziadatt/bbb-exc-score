
import com.actelion.research.chem.MolfileV3Creator;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.SmilesParser;
import com.actelion.research.chem.forcefield.mmff.ForceFieldMMFF94;
import org.openmolecules.chem.conf.gen.ConformerGenerator;
import org.openmolecules.chem.conf.gen.RigidFragmentCache;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class SeededDataWarriorConformers {
  static final int MAX_TORSION_SETS = 1000;
  static final int MAX_ITERATIONS = 10000;
  static final double GRADIENT_TOLERANCE = 0.0001;
  static final double FUNCTION_TOLERANCE = 1.0e-6;

  public static void main(String[] args) throws Exception {
    Path manifest = Path.of(args[0]);
    Path output = Path.of(args[1]);
    Path failures = Path.of(args[2]);
    double dielectric = Double.parseDouble(args[3]);
    RigidFragmentCache.getDefaultInstance().loadDefaultCache();
    ForceFieldMMFF94.initialize(ForceFieldMMFF94.MMFF94SPLUS);
    Map<String,Object> options = new HashMap<>();
    options.put("dielectric constant", dielectric);
    int inputCount = 0, writtenCount = 0, failureCount = 0;

    try (BufferedReader in = Files.newBufferedReader(manifest, StandardCharsets.UTF_8);
         BufferedWriter sdf = Files.newBufferedWriter(output, StandardCharsets.UTF_8);
         BufferedWriter failed = Files.newBufferedWriter(failures, StandardCharsets.UTF_8)) {
      failed.write("dataframe_row\tinput_id\tsmiles\terror\n");
      String line;
      while ((line = in.readLine()) != null) {
        inputCount++;
        String[] f = line.split("\\t", -1);
        String row = decode(f[0]), id = decode(f[1]), smiles = decode(f[2]);
        long seed = Long.parseLong(f[3]);
        String originalProperties = decode(f[4]);
        try {
          StereoMolecule mol = new StereoMolecule();
          new SmilesParser().parse(mol, smiles);
          ConformerGenerator cg = new ConformerGenerator(seed, true);
          if (!cg.initializeConformers(mol, ConformerGenerator.STRATEGY_LIKELY_SYSTEMATIC,
                  MAX_TORSION_SETS, false))
            throw new Exception("Conformer generator could not initialize");
          mol = cg.getNextConformerAsMolecule(mol);
          if (mol == null) throw new Exception("No conformer generated");
          double energy = Double.NaN;
          String minimizationError = "";

          try {
            energy = minimize(mol, options);
          } catch (Exception e) {
            // Match DataWarrior: retain the generated conformer when
            // MMFF94 typing or minimization fails.
            minimizationError = e.getClass().getSimpleName() + ": " + e.getMessage();
          }

          center(mol);
          mol.setName(id.isBlank() ? "row " + row : id);
          sdf.write(new MolfileV3Creator(mol).getMolfile());
          // Restore every original CSV column using its exact original name.
          for (String entry : originalProperties.split("\u001e", -1)) {
            if (entry.isEmpty()) continue;

            int separator = entry.indexOf('\u001f');
            if (separator < 0) continue;

            String columnName = entry.substring(0, separator);
            String columnValue = entry.substring(separator + 1);

            property(sdf, columnName, columnValue);
          }
          property(sdf, "DATAFRAME_ROW", row);
          property(sdf, "INPUT_ID", id);
          property(sdf, "INPUT_SMILES", smiles);
          property(sdf, "CONFORMER_SEED", Long.toString(seed));
          property(sdf, "CONFORMER_ALGORITHM", "systematic, low-energy biased");
          property(sdf, "TORSION_SOURCE", "crystallographic");
          property(sdf, "FORCE_FIELD", "MMFF94s+");
          property(sdf, "DIELECTRIC_CONSTANT", Double.toString(dielectric));
          property(
              sdf,
              "MMFF94_ENERGY",
              Double.isNaN(energy) ? "" : String.format(Locale.ROOT, "%.9g", energy)
          );
          property(sdf, "MINIMIZATION_ERROR", minimizationError);
          sdf.write("$$$$\n");
          writtenCount++;
        } catch (Throwable e) {
          failureCount++;
          failed.write(clean(row)+"\t"+clean(id)+"\t"+clean(smiles)+"\t"
              +clean(e.getClass().getSimpleName()+": "+e.getMessage())+"\n");
        }
      }
    }
    System.out.printf(Locale.ROOT, "Input: %d  Written: %d  Failed: %d%n",
        inputCount, writtenCount, failureCount);
    if (failureCount != 0) System.exit(1);
  }

  static double minimize(StereoMolecule mol, Map<String,Object> options) throws Exception {
    int[] fragmentNo = new int[mol.getAllAtoms()];
    int fragmentCount = mol.getFragmentNumbers(fragmentNo, false, true);
    double energy = Double.NaN;
    if (fragmentCount == 1) {
      ForceFieldMMFF94 ff = new ForceFieldMMFF94(mol, ForceFieldMMFF94.MMFF94SPLUS, options);
      int error = ff.minimise(MAX_ITERATIONS, GRADIENT_TOLERANCE, FUNCTION_TOLERANCE);
      if (error != 0) throw new Exception("MMFF94 error code " + error);
      energy = ff.getTotalEnergy();
    } else {
      StereoMolecule[] fragments = mol.getFragments(fragmentNo, fragmentCount);
      int maxAtoms = 0;
      for (StereoMolecule fragment : fragments) {
        if (fragment.getAllAtoms() > 2) {
          ForceFieldMMFF94 ff = new ForceFieldMMFF94(fragment, ForceFieldMMFF94.MMFF94SPLUS, options);
          int error = ff.minimise(MAX_ITERATIONS, GRADIENT_TOLERANCE, FUNCTION_TOLERANCE);
          if (error != 0) throw new Exception("MMFF94 error code " + error);
          if (maxAtoms < fragment.getAllAtoms()) {
            maxAtoms = fragment.getAllAtoms(); energy = ff.getTotalEnergy();
          }
        }
      }
      int[] atomInFragment = new int[fragmentCount];
      for (int atom=0; atom<fragmentNo.length; atom++) {
        int fragment = fragmentNo[atom], sourceAtom = atomInFragment[fragment]++;
        mol.setAtomX(atom, fragments[fragment].getAtomX(sourceAtom));
        mol.setAtomY(atom, fragments[fragment].getAtomY(sourceAtom));
        mol.setAtomZ(atom, fragments[fragment].getAtomZ(sourceAtom));
      }
    }
    return energy;
  }

  static void center(StereoMolecule mol) {
    double x=0, y=0, z=0;
    for (int a=0; a<mol.getAllAtoms(); a++) { x+=mol.getAtomX(a); y+=mol.getAtomY(a); z+=mol.getAtomZ(a); }
    x/=mol.getAllAtoms(); y/=mol.getAllAtoms(); z/=mol.getAllAtoms();
    for (int a=0; a<mol.getAllAtoms(); a++) {
      mol.setAtomX(a,mol.getAtomX(a)-x); mol.setAtomY(a,mol.getAtomY(a)-y); mol.setAtomZ(a,mol.getAtomZ(a)-z);
    }
  }

  static void property(BufferedWriter w, String name, String value) throws Exception {
    w.write(">  <"+name+">\n"+clean(value)+"\n\n");
  }
  static String decode(String s) { return new String(Base64.getDecoder().decode(s), StandardCharsets.UTF_8); }
  static String clean(String s) { return s == null ? "" : s.replace('\t',' ').replace('\r',' ').replace('\n',' '); }
}
