from logging import (
    getLogger
)
from os import (
    environ,
    listdir,
    walk
)

from flask import (
    Flask
)
from flask_dropzone import (
    Dropzone
)
from flask_socketio import (
    emit,
    SocketIO
)

##### Define global variables #####
ALLOWED_EXTENSIONS = {'tiff', 'tif', 'jpeg', 'jpg', 'png'}  # For uploaded images #
DOWNLOAD_RESULTS = 'VesselExpress_results.zip'  # File name that will be downloaded #
INTSET = {'denoise', 'ball_radius', 'artifact_size', 'block_size', '3D', 'render', 'sigma_steps',
          'image_resolution_x', 'image_resolution_y'}  # Parameters with int values #
RENDER_FILE = None  # Will be used to serve a render image to render.html #
SKIP_CATEGORIES = {'imgFolder', 'skeletonization', 'segmentation', 'rendering_binary',
                   'rendering_skeleton'}  # Skip these #
UPLOAD_FOLDER = 'VesselExpress/data'  # Pipeline output folder #

##### Define app #####
environ["WERKZEUG_RUN_MAIN"] = ""
app = Flask(
    __name__,
    static_folder='../static',
    template_folder='../templates'
)

##### Initialise Sockets #####
# Logging = True to see Websocket packets #
# cors_allowed_origins ='*' to allow all urls #
# async_ mode is for servers like gunicorn that allow threading #
socketio = SocketIO(app, cors_allowed_origins='*', async_mode=None,
                    logger=False, engineio_logger=False)
log = getLogger('werkzeug')
log.disabled = True

##### Import and register views #####
from home import (
    home,
    download,
    favicon,
    render
)

app.register_blueprint(home)  # Main window #
app.register_blueprint(download)  # Download API #
app.register_blueprint(favicon)  # Favicon #
app.register_blueprint(render)  # Render window #

##### Initialise misc #####
dropzone = Dropzone(app)  # Dropzone <- Fileupload #

##### Define Key and upload folder #####
app.secret_key = b'2545d2d66e85c534f055ffa65bf58b91bc1f0770ff71143cbc33a4c06e62dc19'  # Do not expose this #
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


##### Socket connect and disconnect responses #####
@socketio.on('get_log', namespace='/log')
def communicate_log():  # Communicates with Websocket '/log' when receiving 'get_log' trigger #
    try:  # If log present return it to the console window #
        with open(r'VesselExpress/workflow/.snakemake/log/' +
                  listdir('VesselExpress/workflow/.snakemake/log')[-1], 'r') as logfile:
            processed_log = ''.join([line for line in logfile])[:-1]  # [:-1] to align blinking marker behind it #
        emit('newlog', {'data': processed_log})  # Sends signal named newlog to socket with the log #
    except:
        emit('newlog', {'data': 'Waiting for process...'})


@socketio.on('get_files', namespace='/renderfiles')
def communicate_render():  # Communicates with Websocket '/renderfiles' when receiving 'get_files' trigger #
    rendered_files = []
    for dir, _, files in walk(app.config['UPLOAD_FOLDER']):  # Iterates output folder for .glb files #
        if 'VesselExpress/data\\' in dir or 'VesselExpress/data/' in dir:
            # saves the name of the glb file in a list without _ and .glb #
            [rendered_files.append([
                file[:-4].replace('Binary_', 'Binary: ').replace('Skeleton_', 'Skeleton: ')
            ]) for file in files if file not in rendered_files and file.endswith('.glb')]
    if len(rendered_files) > 0:  # glb files present? -> send the names to the socket #
        emit('newfiles', {'data': rendered_files}, namespace='/renderfiles')
    else:
        emit('nonewfiles', {'data': 'Currently no rendered images available.'})


if __name__ == '__main__':
    socketio.run(app=app, host='0.0.0.0')
