from os import (
    path
)

from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for
)

##### Define blueprints #####
home = Blueprint('home', __name__)
download = Blueprint('', __name__)
favicon = Blueprint('favicon', __name__)
render = Blueprint('render', __name__)

##### Import functions #####
import utils


##### Define views #####
@download.route('/uploads/<name>')
def download_file(name):
    '''Intitiates download of the file in variable name if called.

    :param name: str, filename
    :return: download process
    '''
    return send_from_directory(path.join('../..', current_app.config["UPLOAD_FOLDER"]), name)


@favicon.route('/favicon.ico')
def get_favicon():
    '''Serves favicon to frontend.

    :return: favicon
    '''
    return send_from_directory(path.join(favicon.root_path, '../static'), 'image/favicon.ico',
                               mimetype='image/vnd.microsoft.icon')


@home.route('/home', methods=['GET', 'POST'])
@home.route('/', methods=['GET', 'POST'])
def home_page():
    '''Views function that handles GET and POST requests issued from frontend. Calls main.html.

    :return: main.html: template to render
    config: custom config.json
    '''
    if request.method == 'GET':  # GET requests happen on load/reload #
        current_config = utils.page_load()  # Standard routine on page load/reload #
        return render_template('main.html', config=current_config)  # Serve config to frondend #

    elif request.method == 'POST':
        ##### Standard form requests #####
        if 'json_update' in request.form:
            utils.json_update(request.form)
        elif 'json_reset' in request.form:
            utils.json_reset()
        elif 'download_image' in request.form or 'download_statistics' in request.form:
            all_results = 1 if 'download_image' in request.form else 0  # Whether to zip all files or only statistics #
            filename = utils.download_images(all_results)
            if filename:
                return redirect(url_for('download_file', name=filename))  # Send file to download API #
        elif 'clear_files' in request.form:
            utils.clear_files()
        ##### Dropzone request #####
        if len(request.files) != 0:  # Also an AJAX request but specifically issued from the Dropzone on main.html #
            utils.upload_images(request.files)  # Send files to be uploaded to UPLOAD_FOLDER #

        ##### AJAX form requests #####
        if request.get_json() != None:
            if 'request' in request.get_json()[0]:
                if request.get_json()[0]['request'] == 'start_pipeline':
                    # Get config that are currently shown in the sidebar #
                    # utils.json_update(request.get_json()[0], from_run_pipeline=True)
                    utils.run_pipeline()
                elif request.get_json()[0]['request'] == 'get_render_preview':
                    # Pass name to global variable <- Essential to expose it to render.html #
                    global RENDER_FILE
                    RENDER_FILE = request.get_json()[0]['file_name']
        return redirect(request.url)


@render.route('/render', methods=['GET', 'POST'])
def render_viewer():  # Is called when render.html is initialised in a tab #
    return render_template('render.html', file_name=RENDER_FILE)


@render.route('/uploads/<path:render_filename>')
def render_serve_file(render_filename):  # Is called during render.html initialisation #
    '''Render-Image serve API. Delivers image to render.html.

    :param render_filename: str, filename
    :return: render.glb
    '''
    # Transform the button text back to the initial filenames in the upload folder #
    render_filename = render_filename.replace('render/', '')
    if 'Binary: ' in render_filename:
        render_filename = render_filename.replace('Binary: ', 'Binary_')
        dir_folder = render_filename.replace('Binary_', '')
    elif 'Skeleton: ' in render_filename:
        render_filename = render_filename.replace('Skeleton: ', 'Skeleton_')
        dir_folder = render_filename.replace('Skeleton_', '')
    render_filename = render_filename + '.glb'
    # Expose the file to render.html to be opened in the model viewer #
    return send_from_directory(path.join('../..', current_app.config["UPLOAD_FOLDER"], dir_folder),
                               render_filename, as_attachment=True)
