"""
Download and extract video files from repository
"""
import os
import sys
import tarfile
from six.moves.urllib.request import urlretrieve

last_percent_reported = None
# Code from Udacity deep learning course

def sync(directory, archive_url, archive_size=None):
    """
    If necessary, download and extract videos from web archive
    :param archive_size: number of Bytes expected for archive
    :param archive_url: full url of the archive
    :param directory: directory in which the videos should be extracted
    :return: path to directory holding extracted videos
    """
    downloaded_dir, extracted_dir = [os.path.join(directory, d) for d in ['downloads', 'extracted_videos']]

    # create subfolders if necessary
    for subdir in [downloaded_dir, extracted_dir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    # Download archive
    video_data = maybe_download(downloaded_dir, archive_url, archive_size)

    # Extract videos from archive
    video_folder = maybe_extract(video_data, extracted_dir)

    return os.path.join(extracted_dir, video_folder)


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(directory, webarchiveurl, expected_bytes=None, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(directory, os.path.basename(webarchiveurl))
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', webarchiveurl)
        filename, _ = urlretrieve(webarchiveurl, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    print('Download size: ', statinfo.st_size)
    if (expected_bytes is not None):
        if (statinfo.st_size == expected_bytes):
            print('Found and verified', dest_filename)
        else:
            raise Exception(
                'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


def maybe_extract(filename, directory, force=False):
    root = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]  # remove .tar.gz
    if os.path.isdir(os.path.join(directory, root)) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(directory)
        tar.close()
        print('Done!')
    return root
