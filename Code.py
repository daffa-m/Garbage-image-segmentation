from flask import Flask, render_template, request, url_for 
from werkzeug import secure_filename 
from keras.preprocessing.image import ImageDataGenerator 
import tensorflow as tf 
import numpy as np 
import os 
import pspnet
import cv2
from tensorflow.keras import backend as K
import pathlib
from shutil import copyfile

# try: 
#     import shutil 
#     shutil.rmtree('uploaded / image') 
#     % cd uploaded % mkdir image % cd .. 
#     print() 
# except: 
#     pass

model = pspnet.pspnet50(1, (224, 224, 3), 0.0001) 
model.load_weights('weightsasli5003.h5')

model1 = pspnet.pspnet50(1, (224, 224, 3), 0.0001) 
model1.load_weights('weightspreprocessed5003.h5')

app = Flask(__name__) 

app.config['UPLOAD_FOLDER'] = 'static/asli'
app.config['preprocessed'] = 'static/preprocessed1'
app.config['prediksiasli'] = 'static/prediksiasli'
app.config['prediksipreprocessed'] = 'static/prediksipreprocessed'
app.config['annot'] = 'mentah/testannot'
app.config['annot1'] = 'static/gt'

@app.route('/') 
def upload_f(): 
    return render_template('pred.html') 

H, W = 224, 224

def get_image1(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=[H, W])
    img /= 255.
    return img[None, ...]

def get_image2(path):
    gbr = tf.io.read_file(path)
    gbr = tf.image.decode_png(gbr, channels=1)
    gbr = tf.image.resize(gbr, size=[H, W]) > 0
    gbr = tf.cast(gbr, tf.float32)
    return gbr[None, ...]

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def preprocessing(path, path1):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    ret, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #otsu thresholding
    mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
       cv2.THRESH_BINARY,11,2) #mean adaptive thresholding
    ret1, glob = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #global thresholding

    #Penjajaran citra
    im1 = np.atleast_3d(otsu)
    im2 = np.atleast_3d(mean)
    im3 = np.atleast_3d(glob)

    hasilp = np.concatenate((im3, im2, im1), axis=2)
    cv2.imwrite(path1, hasilp)
    return hasilp

def ramalasli(path): 
    img = get_image1(path)
    pred = model.predict(img)[0]
    print(pred) 
    return pred 

def ramalpreprocessed(path): 
    img1 = get_image1(path)
    pred1 = model1.predict(img1)[0]
    print(pred1) 
    return pred1 

@app.route('/uploader', methods = ['GET', 'POST']) 
def upload_file(): 
    if request.method == 'POST': 
        f = request.files['file'] 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        alamat = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        hasil = os.path.join(app.config['prediksiasli'], secure_filename(f.filename))
        hasil1 = os.path.join(app.config['prediksipreprocessed'], secure_filename(f.filename))
        annotalamat = os.path.join(app.config['annot'], secure_filename(f.filename))
        annotalamat1 = os.path.splitext(annotalamat)[0]+'.png'
        annotalamat3 = os.path.join(app.config['annot1'], secure_filename(f.filename))
        annotalamat4 = os.path.splitext(annotalamat3)[0]+'.png'
        prealamat = os.path.join(app.config['preprocessed'], secure_filename(f.filename))
        prealamat1 = os.path.splitext(prealamat)[0]+'.png'
        pre = preprocessing(alamat, prealamat1)
        annot = get_image2(annotalamat1)
        val = ramalasli(alamat)
        val1 = ramalpreprocessed(prealamat1)
        dice = dice_coef(np.squeeze(annot), np.squeeze(val))
        dice1 = dice_coef(np.squeeze(annot), np.squeeze(val1))
        cv2.imwrite(hasil, 255*val)
        cv2.imwrite(hasil1, 255*val1)
        copyfile(annotalamat1, annotalamat4)
        alamat1 = pathlib.Path(alamat)
        palamat = pathlib.Path(*alamat1.parts[1:])
        prealamat1 = pathlib.Path(prealamat1)
        ppreprocessed = pathlib.Path(*prealamat1.parts[1:])
        annotalamat2 = pathlib.Path(annotalamat4)
        pgt = pathlib.Path(*annotalamat2.parts[1:])
        hasil = pathlib.Path(hasil)
        phasli = pathlib.Path(*hasil.parts[1:])
        hasil1 = pathlib.Path(hasil1)
        phpreprocessed = pathlib.Path(*hasil1.parts[1:])
        print (palamat)
                  
        return render_template('cards-gallery1.html',asli = palamat, preprocessed = ppreprocessed, gt = pgt, hasli = phasli, hpreprocessed = phpreprocessed, akurasi = format(dice), akurasi1 = format(dice1)) 
    
if __name__ == '__main__': 
    app.run() 
