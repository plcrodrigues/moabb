import mne
from moabb.datasets.base import BaseDataset
from moabb.datasets import download as dl
import os
import glob
import zipfile
import yaml
from scipy.io import loadmat
import numpy as np
import numpy as np
from scipy.io import loadmat
from distutils.dir_util import copy_tree
import shutil

BI2012_URL = 'https://zenodo.org/record/2649069/files/'
BI2013_URL = 'https://zenodo.org/record/2669187/files/'
BI2014a_URL = 'https://zenodo.org/record/3266223/files/'
BI2015a_URL = 'https://zenodo.org/record/3266930/files/'


class BrainInvaders2012(BaseDataset):
    '''
    We describe the experimental procedures for a dataset that we have made
    publicly available at https://doi.org/10.5281/zenodo.2649006 in mat and
    csv formats. This dataset contains electroencephalographic (EEG)
    recordings of 25 subjects testing the Brain Invaders (Congedo, 2011), a
    visual P300 Brain-Computer Interface inspired by the famous vintage video
    game Space Invaders (Taito, Tokyo, Japan). The visual P300 is an
    event-related potential elicited by a visual stimulation, peaking 240-600
    ms after stimulus onset. EEG data were recorded by 16 electrodes in an
    experiment that took place in the GIPSA-lab, Grenoble, France, in 2012
    (Van Veen, 2013 and Congedo, 2013). A full description of the experiment
    is available https://hal.archives-ouvertes.fr/hal-02126068.
    Python code for manipulating the data is available at
    https://github.com/plcrodrigues/py.BI.EEG.2012-GIPSA.
    The ID of this dataset is
    BI.EEG.2012-GIPSA.

    **Full description of the experiment and dataset**
    https://hal.archives-ouvertes.fr/hal-02126068

    **Link to the data**
    https://doi.org/10.5281/zenodo.2649006

    **Authors**
    Principal Investigator: B.Sc. Gijsbrecht Franciscus Petrus Van Veen
    Technical Supervisors: Ph.D. Alexandre Barachant, Eng. Anton Andreev,
    Eng. Grégoire Cattan, Eng. Pedro. L. C. Rodrigues
    Scientific Supervisor: Ph.D. Marco Congedo

    **ID of the dataset**
    BI.EEG.2012-GIPSA
    '''

    def __init__(
            self,
            Training=True,
            Online=False):
        super().__init__(
            subjects=list(range(1, 25 + 1)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code='Brain Invaders 2012',
            interval=[0, 1],
            paradigm='p300',
            doi='10.5281/zenodo.2649006')

        self.training = Training
        self.online = Online

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        sessions = {}
        for file_path in file_path_list:

            session_name = 'session_1'
            condition = file_path.split(
                '_')[-1].split('.')[0].split(os.sep)[-1]

            if condition == 'training':
                run_name = 'run_1'
            else:
                run_name = 'run_2'

            chnames = ['F7',
                       'F3',
                       'Fz',
                       'F4',
                       'F8',
                       'T7',
                       'C3',
                       'Cz',
                       'C4',
                       'T8',
                       'P7',
                       'P3',
                       'Pz',
                       'P4',
                       'P8',
                       'O1',
                       'O2',
                       'STI 014']
            chtypes = ['eeg'] * 17 + ['stim']

            X = loadmat(file_path)[condition].T
            S = X[1:18, :]
            stim = (X[18, :] + X[19, :])[None, :]
            X = np.concatenate([S, stim])

            info = mne.create_info(ch_names=chnames, sfreq=128,
                                   ch_types=chtypes, montage='standard_1020',
                                   verbose=False)
            raw = mne.io.RawArray(data=X, info=info, verbose=False)

            # get rid of the Fz channel (it is the ground)
            raw.info['bads'] = ['Fz']
            raw.pick_types(eeg=True, stim=True)

            sessions[session_name] = {}
            sessions[session_name][run_name] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = BI2012_URL + 'subject_' + str(subject).zfill(2) + '.zip'
        path_zip = dl.data_path(url, 'BRAININVADERS2012')
        path_folder = path_zip.strip(
            'subject_' + str(subject).zfill(2) + '.zip')

        # check if has to unzip
        if not(os.path.isdir(path_folder + 'subject{:d}/'.format(subject))):
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        subject_paths = []

        # filter the data regarding the experimental conditions
        if self.training:
            subject_paths.append(
                path_folder +
                'subject_' +
                str(subject).zfill(2) +
                '/training.mat')
        if self.online:
            subject_paths.append(
                path_folder +
                'subject_' +
                str(subject).zfill(2) +
                '/online.mat')

        return subject_paths


class BrainInvaders2013(BaseDataset):
    '''
    We describe the experimental procedures for a dataset that we have made
    publicly available at https://doi.org/10.5281/zenodo.1494163 in mat and
    csv formats. This dataset contains electroencephalographic (EEG)
    recordings of 24 subjects doing a visual P300 Brain-Computer Interface
    experiment on PC. The visual P300 is an event-related potential elicited
    by visual stimulation, peaking 240-600 ms after stimulus onset. The
    experiment was designed in order to compare the use of a P300-based
    brain-computer interface on a PC with and without adaptive calibration
    using Riemannian geometry. The brain-computer interface is based on
    electroencephalography (EEG). EEG data were recorded thanks to 16
    electrodes. A full description of the experiment is available at
    https://hal.archives-ouvertes.fr/hal-02103098. Data were recorded during
    an experiment taking place in the GIPSA-lab, Grenoble, France, in
    2013(Congedo, 2013). Python code for manipulating the data is available
    at https://github.com/plcrodrigues/py.BI.EEG.2013-GIPSA.
    The ID of this dataset is BI.EEG.2013-GIPSA.

    **Full description of the experiment and dataset**
    https://hal.archives-ouvertes.fr/hal-02103098

    **Link to the data**
    https://doi.org/10.5281/zenodo.1494163

    **Authors**
    Principal Investigator: B.Sc. Erwan Vaineau, Ph.D. Alexandre Barachant
    Technical Supervisors: Eng. Anton Andreev, Eng. Pedro. L. C. Rodrigues,
    Eng. Grégoire Cattan
    Scientific Supervisor: Ph.D. Marco Congedo

    **ID of the dataset**
    BI.EEG.2013-GIPSA
    '''

    def __init__(
            self,
            NonAdaptive=True,
            Adaptive=False,
            Training=True,
            Online=False):
        super().__init__(
            subjects=list(range(1, 24 + 1)),
            sessions_per_subject='varying',
            events=dict(Target=2, NonTarget=1),
            code='Brain Invaders 2013a',
            interval=[0, 1],
            paradigm='p300',
            doi='10.5281/zenodo.1494163')

        self.adaptive = Adaptive
        self.nonadaptive = NonAdaptive
        self.training = Training
        self.online = Online

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        sessions = {}
        for file_path in file_path_list:

            session_number = file_path.split(os.sep)[-2].strip('Session')
            session_name = 'session_' + session_number
            if session_name not in sessions.keys():
                sessions[session_name] = {}

            run_number = file_path.split(os.sep)[-1]
            run_number = run_number.split('_')[-1]
            run_number = run_number.split('.mat')[0]
            run_name = 'run_' + run_number

            chnames = ['FP1',
                       'FP2',
                       'F5',
                       'AFz',
                       'F6',
                       'T7',
                       'Cz',
                       'T8',
                       'P7',
                       'P3',
                       'Pz',
                       'P4',
                       'P8',
                       'O1',
                       'Oz',
                       'O2',
                       'STI 014']
            chtypes = ['eeg'] * 16 + ['stim']

            X = loadmat(file_path)['data'].T
            # Target=33285, NonTarget=33286
            stim = X[-1,:]
            stim[stim == 33285] = 2
            stim[stim == 33286] = 1

            info = mne.create_info(ch_names=chnames, sfreq=512,
                                   ch_types=chtypes, montage='standard_1020',
                                   verbose=False)
            raw = mne.io.RawArray(data=X, info=info, verbose=False)

            sessions[session_name][run_name] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        if subject in [1, 2, 3, 4, 5, 6, 7]:
            zipname_list = [
                'subject' +
                str(subject).zfill(2) +
                '_session' +
                str(i).zfill(2) +
                '.zip' for i in range(
                    1,
                    8 +
                    1)]
        else:
            zipname_list = ['subject' + str(subject).zfill(2) + '.zip']

        for i, zipname in enumerate(zipname_list):

            url = BI2013_URL + zipname
            path_zip = dl.data_path(url, 'BRAININVADERS2013')
            path_folder = path_zip.strip(zipname)

            # check if has the directory for the subject
            directory = path_folder + 'subject_' + \
                str(subject).zfill(2) + os.sep
            if not(os.path.isdir(directory)):
                os.makedirs(directory)

            if not(os.path.isdir(directory + 'Session' + str(i + 1))):
                print('unzip', path_zip)
                zip_ref = zipfile.ZipFile(path_zip, "r")
                zip_ref.extractall(path_folder)
                os.makedirs(directory + 'Session' + str(i + 1))
                copy_tree(path_zip.strip('.zip'), directory)
                shutil.rmtree(path_zip.strip('.zip'))

        # filter the data regarding the experimental conditions
        meta_file = directory + os.sep + 'meta.yml'
        with open(meta_file, 'r') as stream:
            meta = yaml.load(stream)
        conditions = []
        if self.adaptive:
            conditions = conditions + ['adaptive']
        if self.nonadaptive:
            conditions = conditions + ['nonadaptive']
        types = []
        if self.training:
            types = types + ['training']
        if self.online:
            types = types + ['online']
        filenames = []
        for run in meta['runs']:
            run_condition = run['experimental_condition']
            run_type = run['type']
            if (run_condition in conditions) and (run_type in types):
                filenames = filenames + [run['filename']]

        # list the filepaths for this subject
        subject_paths = []
        for fname in filenames:
            search_path = directory + os.sep
            search_path = search_path + 'Session*'.format(subject)
            search_path = search_path + os.sep + fname.replace('.gdf', '.mat')
            subject_paths = subject_paths + glob.glob(search_path)

        return subject_paths


class BrainInvaders2014a(BaseDataset):
    '''
    This dataset contains electroencephalographic (EEG) recordings of 64
    subjects playing to a visual P300 Brain-Computer Interface (BCI)
    videogame named Brain Invaders. The interface uses the oddball paradigm
    on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed
    pseudo-randomly to elicit the P300 response. EEG data were recorded using
    16 active dry electrodes with up to three game sessions. The experiment
    took place at GIPSA-lab, Grenoble, France, in 2014. A full description of
    the experiment is available at
    https://hal.archives-ouvertes.fr/hal-02171575. Python code for
    manipulating the data is available at
    https://github.com/plcrodrigues/py.BI.EEG.2014a-GIPSA.
    The ID of this
    dataset is bi2014a.
    '''

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 64 + 1)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code='Brain Invaders 2014a',
            interval=[0, 0.8],
            paradigm='p300',
            doi='10.5281/zenodo.3266223')

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)

        sessions = {}
        session_name = 'session_1'
        sessions[session_name] = {}
        run_name = 'run_1'

        chnames = ['FP1',
                   'FP2',
                   'F3',
                   'AFz',
                   'F4',
                   'T7',
                   'Cz',
                   'T8',
                   'P7',
                   'P3',
                   'Pz',
                   'P4',
                   'P8',
                   'O1',
                   'Oz',
                   'O2',
                   'STI 014']
        chtypes = ['eeg'] * 16 + ['stim']

        file_path = file_path_list[0]
        D = loadmat(file_path)['samples'].T

        S = D[1:17, :]
        stim = D[-1, :]
        X = np.concatenate([S, stim[None, :]])

        info = mne.create_info(ch_names=chnames, sfreq=512,
                               ch_types=chtypes, montage='standard_1020',
                               verbose=False)
        raw = mne.io.RawArray(data=X, info=info, verbose=False)

        sessions[session_name][run_name] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = BI2014a_URL + 'subject_' + str(subject).zfill(2) + '.zip'
        path_zip = dl.data_path(url, 'BRAININVADERS2014A')
        path_folder = path_zip.strip(
            'subject_' + str(subject).zfill(2) + '.zip')

        # check if has to unzip
        path_folder_subject = path_folder + \
            'subject_' + str(subject).zfill(2) + os.sep
        if not(os.path.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        subject_paths = []

        # filter the data regarding the experimental conditions
        subject_paths.append(
            path_folder_subject +
            'subject_' +
            str(subject).zfill(2) +
            '.mat')

        return subject_paths


class BrainInvaders2015a(BaseDataset):
    '''
    This dataset contains electroencephalographic (EEG) recordings
    of 43 subjects playing to a visual P300 Brain-Computer Interface (BCI)
    videogame named Brain Invaders. The interface uses the oddball paradigm
    on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed
    pseudo-randomly to elicit the P300 response. EEG data were recorded using
    32 active wet electrodes with three conditions: flash duration 50ms, 80ms
    or 110ms. The experiment took place at GIPSA-lab, Grenoble, France,
    in 2015. A full description of the experiment is available at
    https://hal.archives-ouvertes.fr/hal-02172347. Python code for
    manipulating the data is available at
    https://github.com/plcrodrigues/py.BI.EEG.2015a-GIPSA.
    The ID of this dataset is bi2015a.
    '''

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 43 + 1)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code='Brain Invaders 2015a',
            interval=[0, 0.8],
            paradigm='p300',
            doi='10.5281/zenodo.3266930')

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)

        sessions = {}
        for file_path, session in zip(file_path_list, [1, 2, 3]):

            session_name = 'session_' + str(session)
            sessions[session_name] = {}
            run_name = 'run_1'

            chnames = ['Fp1',
                       'Fp2',
                       'AFz',
                       'F7',
                       'F3',
                       'F4',
                       'F8',
                       'FC5',
                       'FC1',
                       'FC2',
                       'FC6',
                       'T7',
                       'C3',
                       'Cz',
                       'C4',
                       'T8',
                       'CP5',
                       'CP1',
                       'CP2',
                       'CP6',
                       'P7',
                       'P3',
                       'Pz',
                       'P4',
                       'P8',
                       'PO7',
                       'O1',
                       'Oz',
                       'O2',
                       'PO8',
                       'PO9',
                       'PO10',
                       'STI 014']

            chtypes = ['eeg'] * 32 + ['stim']

            D = loadmat(file_path)['DATA'].T
            S = D[1:33, :]
            stim = D[-2, :] + D[-1, :]
            X = np.concatenate([S, stim[None, :]])

            info = mne.create_info(ch_names=chnames, sfreq=512,
                                   ch_types=chtypes, montage='standard_1020',
                                   verbose=False)
            raw = mne.io.RawArray(data=X, info=info, verbose=False)

            sessions[session_name][run_name] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = BI2015a_URL + 'subject_' + str(subject).zfill(2) + '_mat.zip'
        path_zip = dl.data_path(url, 'BRAININVADERS2015A')
        path_folder = path_zip.strip(
            'subject_' + str(subject).zfill(2) + '.zip')

        # check if has to unzip
        path_folder_subject = path_folder + \
            'subject_' + str(subject).zfill(2) + os.sep
        if not(os.path.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        # filter the data regarding the experimental conditions
        subject_paths = []
        for session in [1, 2, 3]:
            subject_paths.append(
                path_folder_subject +
                'subject_' +
                str(subject).zfill(2) +
                '_session_' +
                str(session).zfill(2) +
                '.mat')

        return subject_paths
