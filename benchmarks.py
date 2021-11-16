# benchmarks.py
# erica busch, 6/2020
import numpy as np
import os, glob, sys
from scipy.spatial.distance import cdist
import sklearn.decomposition.PCA as PCA

def vertex_isc(data, data2=None):
    """
    Performs an intersubject correlation of vertex response profiles, comparing each subject's response profiles at each vertex
    to the mean response profile of all the other subjects.

    Parameters:
    ----------
    data: a n_subjects-length list of (timeseries, features) datasets upon which to perform ISC. Used to compute group avg
    data2 (optional): a n_subjects-length list of (timeseries, features) datasets upon which to perform ISC. Compared with the group avg
                      if none than set to be equal to data. default: None

    Returns:
    -------
    all_results: a numpy array of shape (n_subjects, n_features) of ISC values.
    """
    if data2 is None:
        data2 = data
    all_results = np.ndarray((data.shape[0],data.shape[2]), dtype=float)
    all_subjs = np.arange(data.shape[0])
    for v in np.arange(data.shape[2]):
        data_v = data[:,:,v]
        data_v2 = data2[:,:,v]
        # hold out one subject; compare with average of remaining subjects
        for subj, ds in enumerate(data_v2):
            group = np.setdiff1d(all_subjs, subj)
            group_avg = np.mean(data_v[group,:], axis=0).ravel()
            r = np.corrcoef(group_avg, ds.ravel())[0,1]
            all_results[subj, v] = r
    return np.array(all_results)


def dense_connectivity_profile_isc(data):
    """
    Takes the data and creates a vertex-by-vertex full connectivity matrix for each subject, then performs ISC on the
    connectivity profiles.

    Parameters:
    ----------
    data: a n_subjects-length list of (timeseries, features) datasets from which to compute a connectivity matrix.

    Retu
    -------
    all_results: a numpy array of shape (n_subjects, n_features) of ISC values.

    """
    from mvpa2.datasets.base import Dataset
    from mvpa2.mappers.fxy import FxyMapper

    conn_metric = lambda x,y: np.dot(x.samples, y.samples)/x.nsamples
    connectivity_mapper = FxyMapper(conn_metric)
    connectomes = np.ndarray((data.shape[0], data.shape[2], data.shape[2]), dtype=float)
    for i,ds in enumerate(data):
        d = Dataset(ds)
        conn_targets = Dataset(samples=ds.T)
        connectivity_mapper.train(conn_targets)
        connectomes[i]=connectivity_mapper.forward(d)
        del conn_targets,d
    results = vertex_isc(connectomes)
    return results

## all of this runs between subject multivariate time segment classifications
def searchlight_timepoint_clf(data, window_size=5, buffer_size=10, NPROC=16):
    """
    Performs a sliding window between-subject multivariate classification on each time segment.

    Parameters:
    -----------
    data: a n_subjects-length list of (timeseries, features) datasets from which to compute a connectivity matrix.
    window_size: defaults to 5. The number of TRs to be considered in each classification.
    buffer_size: defaults to 10. The number of TRs to be excluded from the classification before and after the window.
    NPROC: defaults to 16. The number of parallel processes you can use.

    Returns:
    --------
    results: a (n_subjects, features) array of classification accuracies.
    """
    from joblib import Parallel, delayed
    searchlights = get_searchlights('b', utils.SEARCHLIGHT_RADIUS)
    results = []
    for test_subj, sub_id in enumerate(utils.subjects):
        train_subj = np.setdiff1d(range(len(utils.subjects)), test_subj)
        ds_train = np.mean(dss[train_subj],axis=0)
        ds_test = dss[test_subj]
        results.append(get_subj_accuracy(sub_id, ds_train, ds_test, searchlights, window_size, buffer_size))
    results = np.stack(results)
    return results

def get_subj_accuracy(subj_id, ds_train, ds_test, searchlights, window_size, buffer_size, NPROC):
    sl_errors,jobs = [],[]
    n_timepoints = ds_train.shape[0]
    for sl in searchlights:
        train_ds_sl = ds_train[:,sl]
        test_ds_sl = ds_test[:,sl]
        jobs.append(delayed(run_clf_job)(train_ds_sl, test_ds_sl, n_timepoints, window_size, buffer_size))
    with Parallel(n_jobs=NPROC) as parallel:
        accuracy = np.array(parallel(jobs))
    return accuracy

def run_clf_job(train_ds_sl, test_ds_sl, n_timepoints, window_size, buffer_size):
    clf_errors=[]
    for t0 in np.arange(n_timepoints - window_size):
        foil_startpoints = get_foil_startpoints(n_timepoints, t0, window_size, buffer_size)
        target_index = foil_startpoints.index(t0)
        # average across all timepoints within the foil segments to get one score per segment, then average across participants
        # spatiotemporal patterns for all foil segments in the SL
        train_ = np.stack([np.ravel(train_ds_sl[t:t+window_size]) for t in foil_startpoints])
        test_ = np.ravel(test_ds_sl[t0: t0+window_size])
        dist = cdist(train_,test_[np.newaxis,:],metric='correlation')
        winner = np.argmin(dist)
        clf_errors.append(int(winner == target_index))
    return np.mean(np.array(clf_errors))

def get_foil_startpoints(n_timepoints, t0, window_size, buffer_size):
    pre_target, post_target = get_foil_boundaries(np.arange(n_timepoints),t0, window_size, buffer_size)
    foil_startpoints = [t0]
    if pre_target:
        foil_startpoints += range(0, pre_target)
    if post_target:
        foil_startpoints += range(post_target, n_timepoints - window_size)
    return sorted(foil_startpoints)

# this returns the final possible start point of a foil segment before our target segment
# and the first possible start point after the target segment
# this will return none if there are no valid foil segments before or after a given startpoint.
def get_foil_boundaries(timepoint_arr, tstart, window_size, buffer_size):
    end_of_first_buffer, start_of_second_buffer = None, None
    if tstart > window_size + buffer_size:
        end_of_first_buffer = np.argmin(abs(timepoint_arr - (tstart - window_size - buffer_size)))
    if (tstart + window_size * 2 + buffer_size) < len(timepoint_arr):
        start_of_second_buffer = np.argmin(abs(timepoint_arr - (tstart + window_size + buffer_size)))
    return end_of_first_buffer, start_of_second_buffer


def representational_geometry(data):
    '''
    Plan for ISC of representation geometry:
    For each fold:
    (1) load data
    (2) separate by participant (29 subj, 443 times, 9000 vert)
        corr --> 29 X 443 X 443 --> get upper triangle
        --> calculate ISC --> 29X29
    Then average across fold
    '''
    n_subj, n_timesteps, n_verts = data.shape
    rdm = np.zeros((n_subj, n_timesteps*(n_timesteps-1)//2))
    ind = np.triu_indices(n_timesteps, k=1)
    for i in range(n_subj):
        rdm_full = 1 - np.corrcoef(data[i].T)
        rdm[i] = rdm_full[ind]
    results = vertex_isc(np.reshape(rdm, (n_subj, n_timesteps*(n_timesteps-1)//2, 1)))
    return results



def spatial_psf_isc(data, surface, dataset, max_rad=4):
    '''
    (1) for each id use qe to get neighbors
    (2) get timeseries of neighbors from data
    (3) get correlation between center and neighbor timeseries
    '''
    n_subj, n_timesteps, n_verts = data.shape
    psfs = np.empty((max_rad, n_subj, n_verts)) # should be rad X subj X vertices
    for r in range(1, max_rad):
        qe = SurfaceRingQueryEngine(inner_radius=r-1, radius=r, surface=surface)
        qe.train(dataset)
        neighbor_time_series = np.zeros_like(data)
        for v in range(n_verts):
            neighbors = qe.query_byid(v)
            neighbor_time_series[:,:,v] = np.mean(data[:,:,neighbors], axis=-1) # subj X timesteps X features
        psfs[r-1,:,:] = vertex_isc(neighbor_time_series, data)
    return psfs

def temporal_psf_isc(data, max_dist=4):
    '''
    (1) for each timestep get neighboring timesteps
    (2) get patterns of neighboring timepoints from data
    (3) get correlation between center and neighbor patterns
    '''
    n_subj, n_timesteps, n_verts = data.shape
    psfs = np.empty((max_dist*2, n_subj, n_timesteps)) # should be subject by distance X 2 (left and right) X subj X TRs (?)
    for d in range(1, max_dist):
        neighbor_patterns = np.zeros_like(data)
        for t in range(d, n_timesteps-d):
            neighbor_patterns[:,t,:] = np.mean(data[:,t-d,:], axis=-1) # subj X timesteps X features
        psfs[max_dist-d,:,:] = vertex_isc(neighbor_patterns, data)
        neighbor_patterns = np.zeros_like(data)
        for t in range(d, n_timesteps-d):
            neighbor_patterns[:,t,:] = np.mean(data[:,t+d,:], axis=-1) # subj X timesteps X features
        psfs[max_dist+d-1,:,:] = vertex_isc(neighbor_patterns, data)
    return psfs

def run_pca(data, data_test=None, rank=50):
    '''
    Use concatenated PCA
    '''
    n_subj, n_timesteps, n_verts = data.shape
    assert rank<n_verts
    pca = PCA(n_components=rank)
    pca.fit(data.reshape(n_subj*n_timesteps, n_verts))
    if data_test is None:
        data_test = data
    low_data = pca.transform(data_test.reshape(n_subj*n_timesteps, n_verts))
    recon_data = pca.inverse_transform(low_data).reshape(n_subj, n_timesteps, n_verts)
    return low_data, recon_data, pca

