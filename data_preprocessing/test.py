import time
import datetime

import numpy as np
from mdtraj.formats import XTCTrajectoryFile
import matplotlib.pyplot as plt
from numpy import ndarray
import os

"""
    Reference: A Machine Learning Approach for the Discovery of Ligand-Specific Functional Mechanisms of GPCRs
    Raw Input data file: .xtc
    -> extract frames and coordinates of atoms from trajectory file
    -> scramble the external degrees of freedom (translation and rotation)
"""

time_spent_in_positional_scrambling = 0
time_spent_in_orientational_scrambling = 0
time_spent_in_embedding = 0
time_spent_in_prime_factorization = 0
time_spent_in_image_creation = 0

def positional_scrambling(frame) -> ndarray:
    """
    scramble the position of each frame (randomly)
    the positional scrambling of frames from a trajectory by moving the centers of mass
    to a randomly sampled coordinate within a sphere of diameter 90 Å (the size of the largest dimension of the receptor
    :param frame: np.ndarray, shape=(n_atoms, 3)
        A two dimensional numpy array, with the cartesian coordinates of each atoms.
    :return:
    """
    start = time.time()

    # calculate the center of mass of the frame
    com = frame.mean(axis=0)
    # generate a random point within a sphere of diameter 90 Å
    radius = 45.0  # radius is half of diameter
    random_point = np.random.uniform(-radius, radius, size=3)
    # move the center of mass to the random point
    translation_vector = random_point - com
    frame += translation_vector

    end = time.time()
    global time_spent_in_positional_scrambling
    time_spent_in_positional_scrambling += end - start

    return frame

def orientational_scrambling(frame) -> ndarray:
    """
    scramble the orientation of each frame (randomly)
    by aligning a vector defined by two arbitrarily chosen atoms (preferably on an axis connecting the intracellular and extracellular ends of the GPCR) to a random unit vector in spheri
    :param frame: np.ndarray, shape=(n_atoms, 3)
        A two dimensional numpy array, with the cartesian coordinates of each atoms.
    :return:
    """
    start = time.time()

    # Define two arbitrary atoms, here we choose the first two atoms in the frame
    atom1 = frame[0]
    atom2 = frame[1]
    # Define the vector between the two atoms
    vector = atom2 - atom1
    vector /= np.linalg.norm(vector)  # Normalize the vector
    # Generate a random unit vector
    random_vector = np.random.normal(size=3)
    random_vector /= np.linalg.norm(random_vector)  # Normalize the random vector
    # Compute the rotation matrix using the axis-angle representation
    v = np.cross(vector, random_vector)
    c = np.dot(vector, random_vector)
    s = np.linalg.norm(v)
    if s == 0:
        return frame  # No rotation needed if vectors are parallel
    v /= s  # Normalize the rotation axis
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    # Apply the rotation to all atoms in the frame
    frame = np.dot(frame, rotation_matrix.T)

    end = time.time()
    global time_spent_in_orientational_scrambling
    time_spent_in_orientational_scrambling += end - start
    return frame


def scramble_external_dof(frame) -> ndarray:
    """
    The scrambling protocol applied to the data prior to submission to the neural network (NN) is
an unbiasing step in which the position of each frame and its orientation are scrambled (randomly;
see Figure 3 for more information on the trajectory scrambling). This is undertaken in order to eliminate
from consideration by the NN any differences among frames that originate not from the time-dependent
molecular dynamics, but from changes in position or orientation of the ligand-GPCR complex. Thus,
the scrambling directs the NN algorithm to consider only the intramolecular changes of the protein
induced by the ligands. This scrambling is introduced in our protocol to achieve the same unbiasing
that is attained in image classification tasks by random orientation of objects in pictures (which forces
the object recognition neural networks to understand the shapes and colors of objects, independent of
their background and orientation).
-- A Machine Learning Approach for the Discovery of Ligand-Specific Functional Mechanisms of GPCRs
    :param frame: np.ndarray, shape=(n_atoms, 3)
        A two dimensional numpy array, with the cartesian coordinates of each atoms.
    :return:
    """

    frame = positional_scrambling(frame)
    frame = orientational_scrambling(frame)
    return frame  # No actual scrambling implemented


def primeFactor(n):
    ans = []
    start = time.time()

    # Loop from 2 to n
    for i in range(2, n + 1):

        # n % i == 0 means n is divisible by i
        while n % i == 0 and n > 0:
            ans.append(i)

            # divide n by i to remove this factor
            n = n // i
    end = time.time()
    global time_spent_in_prime_factorization
    time_spent_in_prime_factorization += end - start
    return ans



def calculate_image_size(n_pixels):
    factors = primeFactor(n_pixels)
    width = height = 1
    while factors:
        factor = factors.pop()
        if width <= height:
            width *= factor
        else:
            height *= factor
    return width, height

def create_image(frame, width, height) -> ndarray:
    """
    Create an image from the frame (xyz coordinates)
    :param frame:
    :param width:
    :param height:
    :return:
    """
    start = time.time()
    # Normalize all channels at once using vectorized operations
    frame_normalized = (frame - frame.min(axis=0)) / (np.ptp(frame, axis=0)) * 255
    frame_normalized = frame_normalized.astype(np.uint8)

    # Reshape directly to image dimensions
    image = frame_normalized.reshape(height, width, 3)

    end = time.time()
    global time_spent_in_image_creation
    time_spent_in_image_creation += end - start

    return image


def frame_embedding(frame) -> ndarray:
    """
    Embed this frame (xyz coordinates) into an image representation (xyz -> rgb)

    :param frame:
    :return:
    """
    start = time.time()
    frame_size = frame.shape[0] # = n_atoms
    width, height = calculate_image_size(frame_size)
    frame = scramble_external_dof(frame)
    image = create_image(frame, width, height)
    end = time.time()
    global time_spent_in_embedding
    time_spent_in_embedding += end - start


    return image

def create_plot(frame):
    plt.figure()
    plt.scatter(frame[:, 0], frame[:, 1], s=1)
    plt.title('Frame after scrambling')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.axis('equal')

def progress(current, total, start_time, bar_length=40, ):
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '=' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    ending = '\n' if current == total else '\r'
    elapsed = time.time() - start_time
    # elapsed time in hh:mm:ss
    elapsed = str(datetime.timedelta(seconds=int(elapsed)))
    print(f'Progress: {current}/{total} [{arrow}{padding}] {int(fraction*100)}% Elapsed Time: {elapsed}s', end=ending)


if __name__ == '__main__':
    # input trajectory file
    enzyme_name = 'DPP8'
    ligand_selective = 'DPP9selective'
    traj_name = f"traj-{enzyme_name}_Cpd42_{ligand_selective}"
    extension = '.xtc'
    data_directory = '../raw_data/Trajectory data/'

    # filepath to the trajectory file
    filepath_input_traj = f"{data_directory}{traj_name}{extension}"

    embedded_data_dir = '../embedded_data/'
    embedded_traj_data_dir =  f"{embedded_data_dir}{traj_name}"
    # create the directory if it does not exist
    if not os.path.exists(embedded_traj_data_dir):
        os.makedirs(embedded_traj_data_dir)




    with XTCTrajectoryFile(filepath_input_traj) as traj:
        xyz, frame_time, step, box = traj.read()

        current_frame_idx = 0
        nr_frames = xyz.shape[0]
        start_time = time.time()
        # output progress every x frames
        polling_rate = 10
        next_poll = 0

        print(
            f"Processing trajectory: {filepath_input_traj}\n",
            f"Output directory for embedded frames: {embedded_traj_data_dir}\n",
        )

        # loop over all frames and apply the scrambling and embedding
        for i in range(xyz.shape[0]):
            # time the processing of a single frame
            start = time.time()
            frame = xyz[i]  # take the i-th frame
            image = frame_embedding(frame) # embed the frame into an image
            embedded_frame_time = time.time()
            image_filename = f"{embedded_traj_data_dir}/frame_{i:05d}.png"
            # save the image (ndarray)
            plt.imsave(image_filename, image)

            end = time.time()
            current_frame_idx += 1
            if current_frame_idx >= next_poll:
                next_poll += polling_rate
                progress(current_frame_idx, nr_frames, start_time)

    print("Done")
    print(f"""Timing statistics:
    Time spent in positional scrambling: {time_spent_in_positional_scrambling} seconds
    Time spent in orientational scrambling: {time_spent_in_orientational_scrambling} seconds
    Time spent in embedding: {time_spent_in_embedding} seconds
    Time spent in prime factorization: {time_spent_in_prime_factorization} seconds
    Time spent in image creation: {time_spent_in_image_creation} seconds
    """)


