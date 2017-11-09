from flask import Flask, send_file, render_template, request, redirect, url_for, flash, send_from_directory
import stitch_images
from werkzeug.utils import secure_filename
import os

# UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])


app = Flask(__name__)

# UPLOAD_FOLDER = os.path.basename('uploads')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
@app.route("/home")
@app.route("/index")
def home():
    return render_template('home.html')

@app.route("/get_stitched_urls")
def stitch_from_urls():
    return render_template('stitch_urls.html')

@app.route("/get_default_images/<dest>")
def hello2(dest):
    stitch_images.get_default_images(str(dest))
    return render_template('images2.html')

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            image1 = request.form['image1']
            image2 = request.form['image2']

            stitch_images.get_image_by_url(image1,image2)
            return render_template('images2.html')
        except:
            # con.rollback()
            msg = "error in insert operation"
            return msg
    else:
        return redirect(url_for('home'))




# @app.route("/image/<dest>")
# def abcd(dest):
#     result_image = stitch_images.get_image(str(dest))
#     filename = 'static/result_image.jpg'
#     # return 'Hello WOrlddds!'
#     return send_file(filename, mimetype='image/jpeg')


@app.route("/clear_static")
def clear_uploads():
    folder = 'static'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    return redirect(url_for('home'))

@app.route("/uploaded2")
def uploaded():
    return 'Uploaded!'

# @app.route("/imagea")
# def hello3():
#     # result_image = stitch_images.get_image(str(dest))
#     return render_template('images.html')
#
# @app.route("/imageb")
# def hello4():
#     # result_image = stitch_images.get_image(str(dest))
#     return render_template('images2.html')

# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit a empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('home'))
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <p><input type=file name=file multiple>
#          <input type=submit value=Upload>
#     </form>
#     '''

@app.route('/upload2', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        # check if the post request has the file part
        files = []
        files.append(request.files['file1'])
        files.append(request.files['file2'])

        # first_file = request.files['file1']
        # second_file = request.files['file2']

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('home'))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file1>
      <p><input type=file name=file2>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)



@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=8000
            )
