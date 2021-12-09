from os import (
    listdir,
    path
)

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for
)
from werkzeug.utils import (
    secure_filename
)

##### Define blueprints #####
home = Blueprint('home', __name__)
download = Blueprint('', __name__)
favicon = Blueprint('favicon', __name__)
render = Blueprint('render', __name__)

##### Import functions #####
import utils
from app import (
    ALLOWED_EXTENSIONS,
    DOWNLOAD_RESULTS
)


##### Define views #####
@download.route('/uploads/<string:name>')
def download_file(name):
    '''Intitiates download of the file in the variable name if called.

    :param name: str, filename
    :return: download process
    '''
    return send_from_directory(path.join('../..', current_app.config["UPLOAD_FOLDER"]), secure_filename(name))


@favicon.route('/favicon.ico')
def get_favicon():
    '''Serves favicon to frontend.

    :return: favicon
    '''
    return send_from_directory(path.join(favicon.root_path, '../static'), 'image/favicon.ico',
                               mimetype='image/vnd.microsoft.icon')


##### Main View #####
@home.route('/home', methods=['GET', 'POST'])
@home.route('/', methods=['GET', 'POST'])
def home_page():
    '''Views function that handles GET and POST requests issued from frontend. Calls main.html.

    :return: main.html: template to render
    config: custom config.json
    '''
    if request.method == 'POST':
        ##### Downloads #####
        if 'download_images' in request.form or 'download_statistics' in request.form:
            return redirect(url_for('download_file', name=DOWNLOAD_RESULTS))  # Send file to download API #
        elif 'download_logs' in request.form:
            filename = utils.download_logs()  # Create zip file #
            if filename:
                return redirect(url_for('download_file', name=filename))  # Send file to download API #

        ##### Dropzone request #####
        if len(request.files) != 0:  # Also an AJAX request but specifically issued from the Dropzone on main.html #
            utils.upload_images(request.files)  # Send files to be uploaded to UPLOAD_FOLDER #

        ##### AJAX form requests #####
        if request.get_json() != None:
            flash_collection = []
            instruction = request.get_json()[0]
            if 'request' in instruction:
                if instruction['request'] == 'clear_files':
                    flash_collection = utils.clear_files(flash_collection)
                elif instruction['request'] == 'download_images' or instruction['request'] == 'download_statistics':
                    if instruction['type'] == 'second_run':
                        return 'True'
                    # Whether to zip all files or only statistics #
                    all_results = 1 if instruction['request'] == 'download_images' else 0
                    flash_collection = utils.download_images(flash_collection=flash_collection,
                                                             firstrun=instruction['type'],
                                                             all_results=all_results)
                elif instruction['request'] == 'download_logs':
                    flash_collection = utils.download_logs(flash_collection=flash_collection, firstrun=True)

                elif instruction['request'] == 'get_render_preview':
                    # Pass name to global variable <- Essential to expose it to render.html #
                    global RENDER_FILE
                    RENDER_FILE = secure_filename(instruction['file_name'])

                elif instruction['request'] == 'reset_config':
                    flash_collection = utils.reset_config(flash_collection)
                elif instruction['request'] == 'start_pipeline':
                    flash_collection = utils.run_pipeline(flash_collection)
                elif instruction['request'] == 'update_config':
                    flash_collection = utils.update_config(instruction['update_config'], flash_collection)
                return jsonify(flash_collection)
    current_config = utils.page_load()  # Standard routine on page load/reload #
    return render_template('main.html', config=current_config)  # Serve config to frontend #


##### Render View #####
@render.route('/render', methods=['GET', 'POST'])
def render_viewer():  # Is called when render.html is initialised in a tab #
    return render_template('render.html', file_name=RENDER_FILE)


@render.route('/render/<string:render_filename>')
def render_serve_file(render_filename):  # Is called during render.html initialisation #
    '''Render-Image serve API. Delivers image to render.html.

    :param render_filename: str, filename
    :return: render.glb
    '''
    dir_folder = render_filename.replace('Binary_', '') if 'Binary_' in render_filename \
        else render_filename.replace('Skeleton_', '')
    # Checks if an image file is in the result folder #
    if any((image.endswith(tuple(ALLOWED_EXTENSIONS))) for image in
           listdir(path.join(current_app.config['UPLOAD_FOLDER'], dir_folder)) if render_filename + '.'):
        # Expose the file to render.html to be opened in the model viewer #
        return send_from_directory(path.join('../..', current_app.config["UPLOAD_FOLDER"], dir_folder),
                                   secure_filename(render_filename + '.glb'))
