import os
def get_audio_path(path, filename):
    return os.path.join(path, 'clips', filename)