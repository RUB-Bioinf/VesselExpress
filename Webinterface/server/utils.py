from json import (
    dump,
    load
)
from multiprocessing import (
    Event,
    Process,
    Queue
)
from os import (
    chdir,
    listdir,
    path,
    remove as osremove,
    rmdir,
    walk
)
from subprocess import (
    run as sbsrun
)
from time import (
    sleep
)
from zipfile import (
    ZipFile
)

from flask import (
    current_app,
    flash
)
from werkzeug.utils import (
    secure_filename
)

from app import (
    ALLOWED_EXTENSIONS,
    DOWNLOAD_RESULTS,
    INTSET,
    SKIP_CATEGORIES
)


def allowed_file(filename):
    '''Analyses if the file is allowed given predetermined extension set in app.py.

    :param filename: str, Name of a file with extension
    :return: bool
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_files():
    '''Clears all files and folders in the upload directory predetermined in app.py.
    Always skips .json files and can skip .zip on demand. Flashes number of deleted files.

    :return: None
    '''
    try:
        if not any(not (files.endswith('.json')) for files in listdir(current_app.config['UPLOAD_FOLDER'])):
            flash('Nothing to delete', 'error')
        else:
            number_of_files, subdirlist = 0, []
            for dir, subdirs, files in walk(current_app.config['UPLOAD_FOLDER']):
                if dir == current_app.config['UPLOAD_FOLDER'] and len(subdirs) > 0:
                    subdirlist = subdirs  # Saves subdir folder names to delete separately #
                for file in files:
                    if not file.endswith('.json'):  # Deletes all files except the config files #
                        osremove(path.join(dir, file))
                        number_of_files += 1
            if len(subdirlist) > 0:  # Delete subdir folders after all files have been counted #
                [rmdir(path.join(current_app.config['UPLOAD_FOLDER'], subdir)) for subdir in subdirlist]
            flash(f'{number_of_files} Files deleted', 'success') if number_of_files > 1 \
                else flash(f'{number_of_files} File deleted', 'success')  # Flash number of deleted files #
    except:
        flash('Files could not be deleted', 'error')


def download_images(all_results):
    '''Archives all files and subdirectories from the upload directory predetermined in app.py. Returns the secured
    filename of the created .zip archive that may be send to the download API.

    :return: Filename, name of the created .zip archive
    '''
    if not any(not (files.endswith(('.json', '.zip'))) for files in listdir(current_app.config['UPLOAD_FOLDER'])):
        return flash('No files to download', 'error')  # If no files other than .json and .zip in folder return this #
    else:
        # Zip everything except files that end with .zip and the config_standard.json #
        with ZipFile(path.join(current_app.config['UPLOAD_FOLDER'], DOWNLOAD_RESULTS), 'w') as zip_filename:
            # Iterating over upload_folder and all subdirs to write them into a zip file #
            for dir, _, files in walk(current_app.config['UPLOAD_FOLDER']):
                # manipulate the directory for the structure INSIDE the zip file so that the zip file  #
                # does not contain 2 empty folders, but only the contents of the UPLOAD_FOLDER #
                zipdir = dir.replace('VesselExpress/data/', '') if 'VesselExpress/data/' in dir \
                    else dir.replace('VesselExpress/data', '')
                if all_results == 1:  # Write all files into the zip file except config_standard and itself (loop) #
                    [zip_filename.write(path.join(dir, filename), path.join(zipdir, filename)) \
                     for filename in files if not (filename.endswith('.zip') or filename == 'config_standard.json')]
                else:  # Zip only .csv and .PNG files <- basic low size files #
                    [zip_filename.write(path.join(dir, filename), path.join(zipdir, filename)) \
                     for filename in files if filename.endswith(('.csv', '.PNG'))]
        return secure_filename(DOWNLOAD_RESULTS)  # Secure filename and send it to download API #


def json_reset():
    '''Overwrites the custom config.json with a predetermined standard config.json file.

    :return: None
    '''
    try:
        with open(current_app.config['UPLOAD_FOLDER'] + '/config_standard.json') as standard_config:
            standard_config = load(standard_config)
        with open(current_app.config['UPLOAD_FOLDER'] + '/config.json', 'w') as config_json:
            dump(standard_config, config_json, indent=2)
        flash('Settings reset successfully', 'success')
    except:
        flash('Default settings could not be restored, please restart the docker image', 'error')


def json_update(form, from_run_pipeline=None):
    '''Updates the custom config.json with the values changed in the sidebar.

    :param form: immutable dict, dict keys mirror the keys in config.json
    :param from_run_pipeline: kwarg, controls execution of flash messages
    :return: None
    '''
    try:
        # Prepare data #
        with open(current_app.config['UPLOAD_FOLDER'] + '/config.json') as config_json:
            current_config = load(config_json)

        input_values = {i: int(form[i]) if i in INTSET else '' if i == 'json_update' else float(form[i]) for i in form}
        # Catch special cases #
        if 'render' not in input_values:  # Needs to be catched here #
            input_values['render'] = 0
        if 'dim_z' not in input_values:  # Keep dim_z value even if not needed #
            input_values['dim_z'] = 0
        if 'render_device' in input_values:
            input_values['render_device'] = "GPU" if input_values['render_device'] == 1 else "CPU"
        input_values['pixel_dimensions'] = ','.join([str(input_values['dim_z']),
                                                     str(input_values['dim_y']),
                                                     str(input_values['dim_x'])])
        # Update config #
        for category in current_config:  # Iterate through the config file to update it #
            if category not in SKIP_CATEGORIES:
                if category in {'3D', 'render'}:  # First level options from the config file #
                    if current_config[category] != input_values[category]:  # Did the value change? #
                        current_config[category] = input_values[category]  # Apply change #
                else:
                    for subcategory in current_config[category]:  # Options with subcategories #
                        if subcategory in input_values:  # Second level options from the config file #
                            if current_config[category][subcategory] != input_values[subcategory]:  # Change? #
                                current_config[category][subcategory] = input_values[subcategory]  # Apply change #
        if not from_run_pipeline:  # Running pipeline is an AJAX request that does NOT refresh the site -> no flash #
            flash('Settings updated successfully', 'success')
    except:
        flash('Settings could not be updated', 'error')
        return
    with open(current_app.config['UPLOAD_FOLDER'] + '/config.json', 'w') as config_json:  # Save the new config #
        dump(current_config, config_json, indent=2)


def page_load():
    '''Accesses custom config.json and modifies dim keys to fit with the sidebar form keys.

    :return: dict, config
    '''
    with open(current_app.config['UPLOAD_FOLDER'] + '/config.json') as config_json:
        current_config = load(config_json)
    for value, subcategory in zip(current_config['graphAnalysis']['pixel_dimensions'].split(','),
                                  ['dim_z', 'dim_y', 'dim_x']):
        current_config['graphAnalysis'][subcategory] = value
    return current_config


def execute_pipeline(queue):
    '''Function called by the Process from run_pipeline.

    :param queue: allows communication with the main Process (the website)
    :return: None
    '''
    chdir('VesselExpress/workflow')
    sbsrun('snakemake --use-conda --cores all --conda-frontend conda', shell=True)
    chdir('../..')
    queue.put(1)


def run_pipeline():
    '''Executes the pipeline. If no image is present in the upload directory predetermined in app.py a test
    image is analysed.

    :return: None
    '''
    if not any((image.endswith(tuple(ALLOWED_EXTENSIONS))) for image in
               listdir(current_app.config['UPLOAD_FOLDER'])):
        flash('No image found!', 'warning')
    else:
        is_pipeline_not_done = True
        try:
            dir = 'VesselExpress/workflow/.snakemake/log'
            for f in listdir(dir):
                osremove(path.join(dir, f))  # Deletes logs if present #
        except:
            pass
        event, queue = Event(), Queue()  # Event synchronizes Processes and Queue allows communication #
        process1 = Process(target=execute_pipeline, args=(queue,))
        process1.start()
        event.set()
        while is_pipeline_not_done == True:
            # Keeps main process in loop while Snakemake pipeline runs #
            # This needs to be done so that the Process can be closed upon completion #
            sleep(2)  # Frequency to check for whether the pipeline is done #
            processed_log = queue.get()  # Tries to receive an update from process1 #
            if processed_log == 1:  # If process1 finished processed_log will be 1 #
                is_pipeline_not_done = False  # Leave the loop #
        process1.join()  # Close process1 #


def upload_images(files):
    '''Saves files in the upload directory predetermined in app.py.

    :param files: immutable dict, request.files that contains file names
    :return: None
    '''
    for key, file in files.items():
        if key.startswith('file') and allowed_file(file.filename):
            file.save(path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
