#from chgnet.model import CHGNetCalculator
from gptff.model.mpredict import ASECalculator
#from upet.calculator import UPETCalculator
#from mace.calculators import mace_mp
#from ase.calculators.mixing import SumCalculator
#from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
#from mattersim.forcefield import MatterSimCalculator
#from sevenn.calculator import SevenNetCalculator

def Calculator ():

    #calc = CHGNetCalculator()
    #calc = CHGNetCalculator.from_file("yourpath")
    #d3_calc = TorchDFTD3Calculator()
    #calc = SumCalculator([chgnet_calc, d3_calc])

    MODEL_WEIGHT = "yourpath/gptff_v1.pth"
    DEVICE = "cuda"
    calc = ASECalculator(MODEL_WEIGHT, DEVICE)

    #calc = UPETCalculator(model="pet-mad-s", version="1.0.2", device="cuda")

    #calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cuda")

    #model_path = "yourpath"
    #calc = mace_mp(model= model_path, dispersion=False, default_dtype="float64", device='cuda')

    #calc = SumCalculator([mace_calc, d3_calc])

    #calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')

    return calc


# ---------------------------------------------------------------------------
# Optional: batched energy adapter.
#
# If your `calc` supports evaluating many structures in one shot, implement
# this function to translate FastTrack's request (a list of ase.Atoms) into
# your backend's native batch API and return a list/array of energies.
#
# If you delete this function (or leave it undefined), FastTrack will fall
# back to the standard ASE point-by-point `atoms.get_potential_energy()` loop.
# ---------------------------------------------------------------------------

def predict_batch(calc, atoms_list, batch_size=128):

    # GPTFF (current default):
    return calc.predict_energies_batched(
        atoms_list, batch_size=batch_size, show_progress=False
    )

    # UPET example (uncomment to use instead of the GPTFF line above):
    # energies = []
    # for i in range(0, len(atoms_list), batch_size):
    #     res = calc.compute_energy(
    #         atoms_list[i:i + batch_size],
    #         compute_forces_and_stresses=False,
    #     )
    #     energies.extend(res["energy"])
    # return energies
