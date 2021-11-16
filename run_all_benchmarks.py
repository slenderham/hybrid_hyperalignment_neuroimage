import os, sys, itertools
import numpy as np
from scipy.sparse import load_npz, save_npz
from mvpa2.base.hdf5 import h5save, h5load
from scipy.stats import zscore
import HA_prep_functions as prep
import hybrid_hyperalignment as h2a
from benchmarks import searchlight_timepoint_clf, vertex_isc, dense_connectivity_profile_is, spatial_psf_isc, temporal_psf_isc, run_pca

os.environ['TMPDIR'] = '/dartfs-hpc/scratch/f002d44/temp'
os.environ['TEMP'] = '/dartfs-hpc/scratch/f002d44/temp'
os.environ['TMP'] = '/dartfs-hpc/scratch/f002d44/temp'
N_LH_NODES_MASKED = 9372
N_JOBS=16
N_BLOCKS=128
TOTAL_NODES=10242
SPARSE_NODES=642
HYPERALIGNMENT_RADIUS=20

def transform_data(data, outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    dss = []
    for d, sub in zip(data, utils.subjects):
        T = np.load(os.path.join(outdir, 'transformations', "subj{}_mapper.npz".format(sub)))
        aligned = np.nan_to_num(zscore((np.asmatrix(ds)*T).A, axis=0))
        dss.append(aligned)
    dss = np.stack(dss) # subj X time X vert
    return dss

# runs benchmarks and saves
def run_benchmarks(fold_basedir, train_ds, test_run, results_dir):
    results_dir, data_dir = os.path.join(fold_basedir, 'results'), os.path.join(fold_basedir, 'data')
    # vertex profile isc
    vert_prof_isc = vertex_isc(dss)
    np.save(os.path.join(results_dir, 'vertex_profile_isc.npy',vert_prof_isc))
    # vertex pattern isc
    vert_patn_isc = vertex_isc(dss_lh.transpose(1,2))
    np.save(os.path.join(results_dir, 'vertex_pattern_isc.npy',vert_patn_isc))
    # conn profile isc
    cnx_results = dense_connectivity_profile_isc(dss)
    np.save(os.path.join(results_dir, 'dense_connectivity_profile_isc.npy'), cnx_results)
    # representational geometry
    rg_results = representational_geometry(dss)
    np.save(os.path.join(results_dir, 'representational_geometry_isc.npy'), rg_results)
    # spatial PSF
    surface = prep.get_freesurfer_surfaces('b')
    dataset_for_qe = utils.get_train_data('b', all_runs)
    spsf = spatial_psf_isc(dss_lh, surface, dss_train)
    np.save(os.path.join(results_dir, 'spatial_psf.npy'), spsf)
    # temporal PSF
    tpsf = temporal_psf_isc(dss)
    np.save(os.path.join(results_dir, 'temporal_psf.npy'), tpsf)

if __name__ == '__main__':
    ha_type = sys.argv[1]
    dataset = sys.argv[2]
    pca_rank = sys.argv[3]
    if dataset == 'budapest':
        import budapest_utils as utils
    elif dataset == 'raiders':
        import raiders_utils as utils
    elif dataset == 'whiplash':
        import whiplash_utils as utils
    else:
        print('dataset must be one of [whiplash,raiders,budapest]')
        sys.exit()
    print('running {a} on {b}'.format(a=ha_type,b=dataset))
    all_runs = np.arange(1, utils.TOT_RUNS+1)

    # check if you specified which run you wanted to hold out.
    # otherwise, iterate through all train/test combos
    if len(sys.argv) > 3:
        test = [int(sys.argv[3])]
        train = np.setdiff1d(all_runs, test)
        train_run_combos = [train]
    else:
        train_run_combos = list(itertools.combinations(all_runs, utils.TOT_RUNS-1))

    for train in train_run_combos:
        test = np.setdiff1d(all_runs, train)
        print('training on runs {r}; testing on run {n}'.format(r=train, n=test))

        # separate testing and training data
        dss_train = utils.get_train_data('b',train)
        dss_test = utils.get_test_data('b', test)
        if pca_rank!='None':
            dss_train = transform_data(dss_train, outdir)
            dss_test_low, dss_test_recon, pca = run_pca(dss_train, dss_test, rank=pca_rank)
            fig = plt.figure()
            plt.plot(pca.explained_variance_ratio_)
            plt.savefig('explained_var')

        # get the node indices to run SL HA, both hemis
        node_indices = np.concatenate(prep.get_node_indices('b', surface_res=TOTAL_NODES))
        # get the surfaces for both hemis
        surface = prep.get_freesurfer_surfaces('b')

        # prepare the connectivity matrices and run HA if we are running CHA
        run_benchmarks('cha', dss_train, test[0], outdir)
        # run response-based searchlight hyperalignment
        run_benchmarks('rha', dss_train, test[0], outdir)
        # run hybrid hyperalignment
        run_benchmarks('h2a', dss_train, test[0], outdir)
