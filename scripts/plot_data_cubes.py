r"""Several scripts to call plot_cube_movie and make movies of GBT data,
simulations, etc.
"""
from plotting import plot_cube as pc

def plot_GBT_mapset(outputdir="/cita/d/www/home/eswitzer/movies/"):
    pc.plot_GBT_maps('GBT_15hr_map_proposal', transverse=True,
                     outputdir=outputdir, skip_noise=True)
    pc.plot_GBT_maps('GBT_15hr_map_proposal', transverse=False,
                     outputdir=outputdir, skip_noise=True)
    pc.plot_GBT_maps('GBT_15hr_map', outputdir=outputdir, transverse=True)
    pc.plot_GBT_maps('GBT_15hr_map', outputdir=outputdir, transverse=False)
    pc.plot_GBT_maps('GBT_22hr_map', outputdir=outputdir, transverse=True)
    pc.plot_GBT_maps('GBT_22hr_map', outputdir=outputdir, transverse=False)
    pc.plot_GBT_maps('GBT_1hr_map', outputdir=outputdir, transverse=True)
    pc.plot_GBT_maps('GBT_1hr_map', outputdir=outputdir, transverse=False)


def plot_GBT_simset(outputdir="/cita/d/www/home/eswitzer/movies/"):
    pc.plot_simulations('sim_15hr', outputdir=outputdir, transverse=True)
    pc.plot_simulations('sim_15hr', outputdir=outputdir, transverse=False)


def plot_GBT_diff_tests(outputdir="/cita/d/www/home/eswitzer/movies/"):
    tcv_15root = "/mnt/raid-project/gmrt/tcv/"
    tcv_15root += "modetest/73_ABCD_all_15_modes_real_maponly/"
    tcv_15map = tcv_15root + "sec_A_15hr_41-90_cleaned_clean_map_I_with_B.npy"
    tcv_15noise = tcv_15root + "sec_A_15hr_41-90_cleaned_noise_inv_I_with_B.npy"
    ers_15root = "/mnt/raid-project/gmrt/eswitzer/GBT/"
    ers_15root += "cleaned_maps/freq_slices_refactor_tests_15modes/"
    ers_15map = ers_15root + "sec_A_15hr_41-90_cleaned_clean_map_I_with_B.npy"
    ers_15noise = ers_15root + "sec_A_15hr_41-90_cleaned_noise_inv_I_with_B.npy"

    pc.plot_difference(tcv_15map, ers_15map, "Temperature (mK)", sigmarange=6.,
                    fractional=False, diff_filename="./map_difference.npy",
                    outputdir=outputdir, transverse=False)

    pc.plot_difference(tcv_15map, ers_15map, "Temperature (mK)", sigmarange=6.,
                    fractional=False, diff_filename="./map_difference.npy",
                    outputdir=outputdir, transverse=True)

    pc.plot_difference(tcv_15noise, ers_15noise, "log inv. covariance", sigmarange=-1.,
                    multiplier=1., logscale=True, fractional=True,
                    diff_filename="./noise_inv_fractional_difference.npy",
                    outputdir=outputdir, transverse=False)

    pc.plot_difference(tcv_15noise, ers_15noise, "log inv. covariance", sigmarange=-1.,
                    multiplier=1., logscale=True, fractional=True,
                    diff_filename="./noise_inv_fractional_difference.npy",
                    outputdir=outputdir, transverse=True)

def plot_sim_scheme(outputdir="/cita/d/www/home/eswitzer/movies/"):
    sim1 = "sim_streaming1.npy"
    sim2 = "sim_streaming2.npy"
    pc.plot_difference(sim1, sim2, "Temperature (mK)", sigmarange=6.,
                    fractional=False, diff_filename="./sim_difference.npy",
                    outputdir=outputdir, transverse=False)

if __name__ == "__main__":
    plot_GBT_mapset()
    #plot_GBT_simset()
    #plot_GBT_diff_tests()
    #plot_sim_scheme()