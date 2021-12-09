from logging import (
    getLogger
)
from os import (
    environ,
)

from flask import (
    Flask
)
from flask_dropzone import (
    Dropzone
)
from flask_socketio import (
    SocketIO,
    emit
)

##### Define global variables #####
ALLOWED_EXTENSIONS = {'tiff', 'tif', 'jpeg', 'jpg', 'png'}  # For uploaded images #
DOWNLOAD_LOGS = 'VesselExpress_logs.zip'  # Filename of the logs #
DOWNLOAD_RESULTS = 'VesselExpress_results.zip'  # Filename of the zipped results #
INTSET = {'denoise', 'ball_radius', 'artifact_size', 'block_size', '3D', 'render', 'sigma_steps',
          'image_resolution_x', 'image_resolution_y'}  # Parameters with int values #
RENDER_FILE = None  # Will be used to serve a render image to render.html #
SKIP_CATEGORIES = {'imgFolder', 'skeletonization', 'segmentation', 'rendering_binary',
                   'rendering_skeleton'}  # Skip these during config update #
UPLOAD_FOLDER = 'VesselExpress/data'  # Pipeline output folder #

##### Define app #####
environ["WERKZEUG_RUN_MAIN"] = ''
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
import utils

app.register_blueprint(home)  # Main window #
app.register_blueprint(download)  # Download API #
app.register_blueprint(favicon)  # Favicon #
app.register_blueprint(render)  # Render window #

##### Initialise misc #####
dropzone = Dropzone(app)  # Dropzone <- Fileupload #

##### Define key and upload folder #####
app.secret_key = b'2545d2d66e85c534f055ffa65bf58b91bc1f0770ff71143cbc33a4c06e62dc19'  # Do not expose this #
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


##### Sockets #####
# They have to be in app or they will not work in Docker #
@socketio.on('get_log', namespace='/log')
def communicate_log():  # Communicates with Websocket '/log' when receiving 'get_log' trigger #
    processed_log = utils.get_last_logfile()
    if processed_log:
        emit('newlog', {'data': processed_log})  # Sends signal named newlog to socket with the log #
    else:
        emit('newlog', {'data': 'Waiting for process...'})


@socketio.on('get_files', namespace='/renderfiles')
def communicate_render():  # Communicates with Websocket '/renderfiles' when receiving 'get_files' trigger #
    rendered_files = utils.get_rendered_files()
    if len(rendered_files) > 0:  # glb files present? -> send the names to the socket #
        emit('newfiles', {'data': rendered_files})
    else:
        emit('nonewfiles', {'data': 'Currently no rendered images available.'})


@socketio.on('get_progress', namespace='/progbar')
def communicate_progbar():  # Communicates with Websocket '/progbar' when receiving 'get_progress' trigger #
    progbar_status = utils.get_progress()
    if 'current' in progbar_status:
        emit('newprogress', {'data': progbar_status})
    else:
        emit('nonewprogress', {'data': 'Currently no progress.'})


##### Start website #####
if __name__ == '__main__':
    socketio.run(app=app, host='0.0.0.0')
