from glob import (
    glob
)
from json import (
    dump,
    load,
    JSONDecodeError
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
    current_app
)
from werkzeug.utils import (
    secure_filename
)

from app import (
    ALLOWED_EXTENSIONS,
    DOWNLOAD_RESULTS,
    DOWNLOAD_LOGS,
    INTSET,
    SKIP_CATEGORIES
)


def allowed_file(filename):
    '''Analyses if the file is allowed given predetermined extension set in app.py.

    :param filename: str, Name of a file with extension
    :return: bool
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_files(flash_collection):
    '''Clears all files and folders in the upload directory predetermined in app.py.
    Always skips .json files and can skip .zip on demand. Flashes number of deleted files.

    :return: None
    '''
    if not any(not (files.endswith('.json')) for files in listdir(current_app.config['UPLOAD_FOLDER'])):
        flash_collection.append({'message': 'No files to delete', 'type': 'error'})
        return flash_collection
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
        if number_of_files > 1:
            flash_collection.append({'message': f'{number_of_files} Files deleted', 'type': 'success'})
        else:
            flash_collection.append({'message': f'{number_of_files} File deleted', 'type': 'success'})
        return flash_collection


def download_images(all_results=None, flash_collection=None, firstrun=None):
    '''Archives all files and subdirectories from the upload directory predetermined in app.py. Returns the secured
    filename of the created .zip archive that may be send to the download API

    :param all_results: Bool, download all results or only statistics
    :param flash_collection: List, flash messages
    :param firstrun: Bool, whether to check for files or zip
    :return: Filename, name of the created .zip archive
    '''
    if not any(not (files.endswith((tuple(ALLOWED_EXTENSIONS) + ('.json', '.zip'))))
               for files in listdir(current_app.config['UPLOAD_FOLDER'])):
        # If no files other than .json and .zip in folder return this #
        if firstrun == "first_run":
            flash_collection.append({'message': 'No results to download', 'type': 'error'})
            return flash_collection
    else:
        # Zip everything except files that end with .zip and the config_standard.json #
        current_filecount = 0
        zip_ignore = len(glob(path.join(current_app.config['UPLOAD_FOLDER'], '*zip')))
        sum_of_files = sum([len(files) for _, _, files in walk(current_app.config['UPLOAD_FOLDER'])]) - 1 - zip_ignore
        with ZipFile(path.join(current_app.config['UPLOAD_FOLDER'], DOWNLOAD_RESULTS), 'w') as zip_filename:
            # Iterating over upload_folder and all subdirs to write them into a zip file #
            for dir, _, files in walk(current_app.config['UPLOAD_FOLDER']):
                # manipulate the directory for the structure INSIDE the zip file so that the zip file  #
                # does not contain 2 empty folders, but only the contents of the UPLOAD_FOLDER #
                zipdir = dir.replace('VesselExpress/data/', '') if 'VesselExpress/data/' in dir \
                    else dir.replace('VesselExpress/data', '')
                if all_results == 1:  # Write all files into the zip file except config_standard and itself (loop) #
                    for filename in files:
                        if not (filename.endswith('.zip') or
                                filename == 'config_standard.json' or
                                filename == 'progbar.json'):
                            zip_filename.write(path.join(dir, filename), path.join(zipdir, filename))
                            # Log progress to json for progress bar #
                            current_filecount += 1
                            progbar = {'current': current_filecount, 'sum': sum_of_files}
                            try:
                                with open(current_app.config['UPLOAD_FOLDER'] + '/progbar.json', 'w') as progbar_json:
                                    dump(progbar, progbar_json, indent=2)
                            except JSONDecodeError:
                                sleep(0.05)
                                with open(current_app.config['UPLOAD_FOLDER'] + '/progbar.json', 'w') as progbar_json:
                                    dump(progbar, progbar_json, indent=2)
                    sleep(0.55)  # makes sure progbar gets updated to 100%
                    osremove(path.join(current_app.config['UPLOAD_FOLDER'], 'progbar.json'))
                else:  # Zip only .csv and .PNG files <- basic low size files #
                    [zip_filename.write(path.join(dir, filename), path.join(zipdir, filename)) \
                     for filename in files if filename.endswith(('.csv', '.PNG'))]
            return 0


def download_logs(flash_collection=None, firstrun=None):
    list_of_logfiles = glob(path.join('.snakemake/log', '*log'))  # List of log files #
    if len(list_of_logfiles) > 0:
        if firstrun:
            return 0
        else:
            with ZipFile(path.join(current_app.config['UPLOAD_FOLDER'], DOWNLOAD_LOGS), 'w') as zip_filename:
                [zip_filename.write(logfile,
                                    logfile.replace('\\', '/').replace('.snakemake/log/', ''))
                 for logfile in list_of_logfiles]
            return secure_filename(DOWNLOAD_LOGS)
    else:
        flash_collection.append({'message': 'No logs to download', 'type': 'error'})
        return flash_collection


def execute_pipeline(queue):
    '''Function called by the Process from run_pipeline.

    :param queue: allows communication with the main Process (the website)
    :return: None
    '''
    sbsrun('snakemake --use-conda --cores all --conda-frontend conda --snakefile "./VesselExpress/workflow/Snakefile"',
           shell=True)
    queue.put(1)


def get_last_logfile():
    '''Identifies the last modified log, processes it to be displayed in the console and returns it.

    :return: processed log
    '''
    try:  # If log present return it to the console window #
        list_of_logfiles = glob(r'.snakemake/log/*log')  # List of log files #
        latest_logfile = max(list_of_logfiles, key=path.getctime)  # Get the latest log #
        with open(latest_logfile, 'r') as logfile:
            processed_log = ''.join([line for line in logfile])[:-1]  # [:-1] to align blinking marker behind it #
    except (IndexError, ValueError):
        processed_log = None
    return processed_log


def get_progress():
    try:
        with open(current_app.config['UPLOAD_FOLDER'] + '/progbar.json', 'r') as progbar_json:
            progbar_status = load(progbar_json)
        return progbar_status
    except FileNotFoundError:
        return []


def get_rendered_files():
    '''Walks through all folders in the upload folder dir and scans for .glb files.

    :param rendered_files: kwarg, list to be filled with .glb filenames.
    :return: list with filesnames.
    '''
    rendered_files = []
    for dir, _, files in walk(current_app.config['UPLOAD_FOLDER']):  # Iterates output folder for .glb files #
        if 'VesselExpress/data\\' in dir or 'VesselExpress/data/' in dir:
            # saves the name of the glb file in a list without _ and .glb #
            [rendered_files.append([
                file[:-4].replace('Binary_', 'Binary: ').replace('Skeleton_', 'Skeleton: ')])
                for file in files if file not in rendered_files and file.endswith('.glb')]
    return rendered_files


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


def reset_config(flash_collection):
    '''Overwrites the custom config.json with a predetermined standard config.json file.

    :return: None
    '''
    with open(current_app.config['UPLOAD_FOLDER'] + '/config_standard.json') as standard_config:
        standard_config = load(standard_config)
    with open(current_app.config['UPLOAD_FOLDER'] + '/config.json', 'w') as config_json:
        dump(standard_config, config_json, indent=2)
    flash_collection.append({'message': 'Settings reset successfully', 'type': 'success'})
    return flash_collection


def run_pipeline(flash_collection):
    '''Executes the pipeline. If no image is present in the upload directory predetermined in app.py a test
    image is analysed.

    :return: None
    '''
    if not any((image.endswith(tuple(ALLOWED_EXTENSIONS))) for image in
               listdir(current_app.config['UPLOAD_FOLDER'])):
        flash_collection.append({'message': 'No image found!', 'type': 'warning'})
        return flash_collection
    else:
        is_pipeline_not_done = True
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
        flash_collection.append({'message': 'The pipeline has finished!', 'type': 'finish'})
        return flash_collection


def update_config(form, flash_collection):
    '''Updates the custom config.json with the values changed in the sidebar.

    :param form: immutable dict, dict keys mirror the keys in config.json
    :param from_run_pipeline: kwarg, controls execution of flash messages
    :return: None
    '''
    # Load old config #
    with open(current_app.config['UPLOAD_FOLDER'] + '/config.json') as config_json:
        current_config = load(config_json)
    input_values = {i: int(form[i]) if i in INTSET else float(form[i]) for i in form}
    # Catch special cases #
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
    flash_collection.append({'message': 'Settings updated successfully', 'type': 'success'})
    with open(current_app.config['UPLOAD_FOLDER'] + '/config.json', 'w') as config_json:  # Save the new config #
        dump(current_config, config_json, indent=2)
    return flash_collection


def upload_images(files):
    '''Saves files in the upload directory predetermined in app.py.

    :param files: immutable dict, request.files that contains file names
    :return: None
    '''
    for key, file in files.items():
        if key.startswith('file') and allowed_file(file.filename):
            file.save(path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
