import os
import zipfile
import tarfile
import requests
import urllib
import progressbar


def download(url, out, check=None):
    """Download and unzip an online archive (.zip, .gz, or .tgz).

    Arguments:
        root (str): Folder to download data to.
        check (str or None): Folder whose existence indicates
            that the dataset has already been downloaded, or
            None to check the existence of root/{cls.name}.

    Returns:
        dataset_path (str): Path to extracted dataset.
    """
    #path = os.path.join(out, cls.name)
    #check = out if check is None else check
    #if not os.path.isdir(out):
        #for url in cls.urls:
    if isinstance(url, tuple):
        url, filename = url
    else:
        filename = os.path.basename(url)
    zpath = os.path.join(out, filename)
    if not os.path.isfile(zpath):
        if not os.path.exists(os.path.dirname(zpath)):
            os.makedirs(os.path.dirname(zpath))
        print('downloading {}'.format(filename))
        download_from_url(url, zpath)
    ext = os.path.splitext(filename)[-1]
    if ext == '.zip':
        with zipfile.ZipFile(zpath, 'r') as zfile:
            print('extracting')
            zfile.extractall(out)
    elif ext in ['.gz', '.tgz']:
        with tarfile.open(zpath, 'r:gz') as tar:
            dirs = [member for member in tar.getmembers()]
            tar.extractall(path=out, members=dirs)
    return zpath



def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        return urllib.request.urlretrieve(url, path)
    print('downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength,
                                  redirect_stdout=True)
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                bar.update()
        bar.finish()
