import numpy as np
from scipy import misc as sm
import shutil
import scipy.io as sio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import skimage.exposure as sk
import cv2
import argparse
import pytiff
import tifffile
import tensorflow as tf
from skimage.morphology import *
from skimage.exposure import rescale_intensity
from skimage.segmentation import chan_vese, find_boundaries, morphological_chan_vese
from skimage.measure import regionprops,label, find_contours
from skimage.transform import resize
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import peak_local_max,blob_log
from skimage.color import gray2rgb as gray2rgb
import skimage.io as skio
from scipy.ndimage.morphology import binary_fill_holes
from skimage import img_as_bool
from skimage.draw import circle_perimeter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage import gaussian_laplace
from os.path import *
from os import listdir, makedirs, remove



import sys
from typing import Any

#sys.path.insert(0, 'C:\\Users\\Public\\Documents\\ImageScience')
from toolbox.imtools import *
from toolbox.ftools import *
from toolbox.PartitionOfImage import PI2D


def concat3(lst):
		return tf.concat(lst,3)

class UNet2D:
	hp = None # hyper-parameters
	nn = None # network
	tfTraining = None # if training or not (to handle batch norm)
	tfData = None # data placeholder
	Session = None
	DatasetMean = 0
	DatasetStDev = 0

	def setupWithHP(hp):
		UNet2D.setup(hp['imSize'],
					 hp['nChannels'],
					 hp['nClasses'],
					 hp['nOut0'],
					 hp['featMapsFact'],
					 hp['downSampFact'],
					 hp['ks'],
					 hp['nExtraConvs'],
					 hp['stdDev0'],
					 hp['nLayers'],
					 hp['batchSize'])

	def setup(imSize,nChannels,nClasses,nOut0,featMapsFact,downSampFact,kernelSize,nExtraConvs,stdDev0,nDownSampLayers,batchSize):
		UNet2D.hp = {'imSize':imSize,
					 'nClasses':nClasses,
					 'nChannels':nChannels,
					 'nExtraConvs':nExtraConvs,
					 'nLayers':nDownSampLayers,
					 'featMapsFact':featMapsFact,
					 'downSampFact':downSampFact,
					 'ks':kernelSize,
					 'nOut0':nOut0,
					 'stdDev0':stdDev0,
					 'batchSize':batchSize}

		nOutX = [UNet2D.hp['nChannels'],UNet2D.hp['nOut0']]
		dsfX = []
		for i in range(UNet2D.hp['nLayers']):
			nOutX.append(nOutX[-1]*UNet2D.hp['featMapsFact'])
			dsfX.append(UNet2D.hp['downSampFact'])


		# --------------------------------------------------
		# downsampling layer
		# --------------------------------------------------

		with tf.name_scope('placeholders'):
			UNet2D.tfTraining = tf.placeholder(tf.bool, name='training')
			UNet2D.tfData = tf.placeholder("float", shape=[None,UNet2D.hp['imSize'],UNet2D.hp['imSize'],UNet2D.hp['nChannels']],name='data')

		def down_samp_layer(data,index):
			with tf.name_scope('ld%d' % index):
				ldXWeights1 = tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index], nOutX[index+1]], stddev=stdDev0),name='kernel1')
				ldXWeightsExtra = []
				for i in range(nExtraConvs):
					ldXWeightsExtra.append(tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index+1], nOutX[index+1]], stddev=stdDev0),name='kernelExtra%d' % i))
				
				c00 = tf.nn.conv2d(data, ldXWeights1, strides=[1, 1, 1, 1], padding='SAME')
				for i in range(nExtraConvs):
					c00 = tf.nn.conv2d(tf.nn.relu(c00), ldXWeightsExtra[i], strides=[1, 1, 1, 1], padding='SAME')

				ldXWeightsShortcut = tf.Variable(tf.truncated_normal([1, 1, nOutX[index], nOutX[index+1]], stddev=stdDev0),name='shortcutWeights')
				shortcut = tf.nn.conv2d(data, ldXWeightsShortcut, strides=[1, 1, 1, 1], padding='SAME')

				bn = tf.layers.batch_normalization(tf.nn.relu(c00+shortcut), training=UNet2D.tfTraining)

				return tf.nn.max_pool(bn, ksize=[1, dsfX[index], dsfX[index], 1], strides=[1, dsfX[index], dsfX[index], 1], padding='SAME',name='maxpool')

		# --------------------------------------------------
		# bottom layer
		# --------------------------------------------------

		with tf.name_scope('lb'):
			lbWeights1 = tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[UNet2D.hp['nLayers']], nOutX[UNet2D.hp['nLayers']+1]], stddev=stdDev0),name='kernel1')
			def lb(hidden):
				return tf.nn.relu(tf.nn.conv2d(hidden, lbWeights1, strides=[1, 1, 1, 1], padding='SAME'),name='conv')

		# --------------------------------------------------
		# downsampling
		# --------------------------------------------------

		with tf.name_scope('downsampling'):    
			dsX = []
			dsX.append(UNet2D.tfData)

			for i in range(UNet2D.hp['nLayers']):
				dsX.append(down_samp_layer(dsX[i],i))

			b = lb(dsX[UNet2D.hp['nLayers']])

		# --------------------------------------------------
		# upsampling layer
		# --------------------------------------------------

		def up_samp_layer(data,index):
			with tf.name_scope('lu%d' % index):
				luXWeights1    = tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index+1], nOutX[index+2]], stddev=stdDev0),name='kernel1')
				luXWeights2    = tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index]+nOutX[index+1], nOutX[index+1]], stddev=stdDev0),name='kernel2')
				luXWeightsExtra = []
				for i in range(nExtraConvs):
					luXWeightsExtra.append(tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index+1], nOutX[index+1]], stddev=stdDev0),name='kernel2Extra%d' % i))
				
				outSize = UNet2D.hp['imSize']
				for i in range(index):
					outSize /= dsfX[i]
				outSize = int(outSize)

				outputShape = [UNet2D.hp['batchSize'],outSize,outSize,nOutX[index+1]]
				us = tf.nn.relu(tf.nn.conv2d_transpose(data, luXWeights1, outputShape, strides=[1, dsfX[index], dsfX[index], 1], padding='SAME'),name='conv1')
				cc = concat3([dsX[index],us]) 
				cv = tf.nn.relu(tf.nn.conv2d(cc, luXWeights2, strides=[1, 1, 1, 1], padding='SAME'),name='conv2')
				for i in range(nExtraConvs):
					cv = tf.nn.relu(tf.nn.conv2d(cv, luXWeightsExtra[i], strides=[1, 1, 1, 1], padding='SAME'),name='conv2Extra%d' % i)
				return cv

		# --------------------------------------------------
		# final (top) layer
		# --------------------------------------------------

		with tf.name_scope('lt'):
			ltWeights1    = tf.Variable(tf.truncated_normal([1, 1, nOutX[1], nClasses], stddev=stdDev0),name='kernel')
			def lt(hidden):
				return tf.nn.conv2d(hidden, ltWeights1, strides=[1, 1, 1, 1], padding='SAME',name='conv')


		# --------------------------------------------------
		# upsampling
		# --------------------------------------------------

		with tf.name_scope('upsampling'):
			usX = []
			usX.append(b)

			for i in range(UNet2D.hp['nLayers']):
				usX.append(up_samp_layer(usX[i],UNet2D.hp['nLayers']-1-i))

			t = lt(usX[UNet2D.hp['nLayers']])


		sm = tf.nn.softmax(t,-1)
		UNet2D.nn = sm


	def train(imPath,logPath,modelPath,pmPath,nTrain,nValid,nTest,restoreVariables,nSteps,gpuIndex,testPMIndex):
		os.environ['CUDA_VISIBLE_DEVICES']= '%d' % gpuIndex

		outLogPath = logPath
		trainWriterPath = pathjoin(logPath,'Train')
		validWriterPath = pathjoin(logPath,'Valid')
		outModelPath = pathjoin(modelPath,'model.ckpt')
		outPMPath = pmPath
		
		batchSize = UNet2D.hp['batchSize']
		imSize = UNet2D.hp['imSize']
		nChannels = UNet2D.hp['nChannels']
		nClasses = UNet2D.hp['nClasses']

		# --------------------------------------------------
		# data
		# --------------------------------------------------

		Train = np.zeros((nTrain,imSize,imSize,nChannels))
		Valid = np.zeros((nValid,imSize,imSize,nChannels))
		Test = np.zeros((nTest,imSize,imSize,nChannels))
		LTrain = np.zeros((nTrain,imSize,imSize,nClasses))
		LValid = np.zeros((nValid,imSize,imSize,nClasses))
		LTest = np.zeros((nTest,imSize,imSize,nClasses))

		print('loading data, computing mean / st dev')
		if not os.path.exists(modelPath):
			os.makedirs(modelPath)
		if restoreVariables:
			datasetMean = loadData(pathjoin(modelPath,'datasetMean.data'))
			datasetStDev = loadData(pathjoin(modelPath,'datasetStDev.data'))
		else:
			datasetMean = 0
			datasetStDev = 0
			for iSample in range(nTrain+nValid+nTest):
				I = im2double(tifread('%s/I%05d_Img.tif' % (imPath,iSample)))
				datasetMean += np.mean(I)
				datasetStDev += np.std(I)
			datasetMean /= (nTrain+nValid+nTest)
			datasetStDev /= (nTrain+nValid+nTest)
			saveData(datasetMean, pathjoin(modelPath,'datasetMean.data'))
			saveData(datasetStDev, pathjoin(modelPath,'datasetStDev.data'))

		perm = np.arange(nTrain+nValid+nTest)
		np.random.shuffle(perm)

		for iSample in range(0, nTrain):
			path = '%s/I%05d_Img.tif' % (imPath,perm[iSample])
			im = im2double(tifread(path))
			Train[iSample,:,:,0] = (im-datasetMean)/datasetStDev
			path = '%s/I%05d_Ant.tif' % (imPath,perm[iSample])
			im = tifread(path)
			for i in range(nClasses):
				LTrain[iSample,:,:,i] = (im == i+1)

		for iSample in range(0, nValid):
			path = '%s/I%05d_Img.tif' % (imPath,perm[nTrain+iSample])
			im = im2double(tifread(path))
			Valid[iSample,:,:,0] = (im-datasetMean)/datasetStDev
			path = '%s/I%05d_Ant.tif' % (imPath,perm[nTrain+iSample])
			im = tifread(path)
			for i in range(nClasses):
				LValid[iSample,:,:,i] = (im == i+1)

		for iSample in range(0, nTest):
			path = '%s/I%05d_Img.tif' % (imPath,perm[nTrain+nValid+iSample])
			im = im2double(tifread(path))
			Test[iSample,:,:,0] = (im-datasetMean)/datasetStDev
			path = '%s/I%05d_Ant.tif' % (imPath,perm[nTrain+nValid+iSample])
			im = tifread(path)
			for i in range(nClasses):
				LTest[iSample,:,:,i] = (im == i+1)

		# --------------------------------------------------
		# optimization
		# --------------------------------------------------

		tfLabels = tf.placeholder("float", shape=[None,imSize,imSize,nClasses],name='labels')

		globalStep = tf.Variable(0,trainable=False)
		learningRate0 = 0.01
		decaySteps = 1000
		decayRate = 0.95
		learningRate = tf.train.exponential_decay(learningRate0,globalStep,decaySteps,decayRate,staircase=True)

		with tf.name_scope('optim'):
			loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(tfLabels,tf.log(UNet2D.nn)),3))
			updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			# optimizer = tf.train.MomentumOptimizer(1e-3,0.9)
			optimizer = tf.train.MomentumOptimizer(learningRate,0.9)
			# optimizer = tf.train.GradientDescentOptimizer(learningRate)
			with tf.control_dependencies(updateOps):
				optOp = optimizer.minimize(loss,global_step=globalStep)

		with tf.name_scope('eval'):
			error = []
			for iClass in range(nClasses):
				labels0 = tf.reshape(tf.to_int32(tf.slice(tfLabels,[0,0,0,iClass],[-1,-1,-1,1])),[batchSize,imSize,imSize])
				predict0 = tf.reshape(tf.to_int32(tf.equal(tf.argmax(UNet2D.nn,3),iClass)),[batchSize,imSize,imSize])
				correct = tf.multiply(labels0,predict0)
				nCorrect0 = tf.reduce_sum(correct)
				nLabels0 = tf.reduce_sum(labels0)
				error.append(1-tf.to_float(nCorrect0)/tf.to_float(nLabels0))
			errors = tf.tuple(error)

		# --------------------------------------------------
		# inspection
		# --------------------------------------------------

		with tf.name_scope('scalars'):
			tf.summary.scalar('avg_cross_entropy', loss)
			for iClass in range(nClasses):
				tf.summary.scalar('avg_pixel_error_%d' % iClass, error[iClass])
			tf.summary.scalar('learning_rate', learningRate)
		with tf.name_scope('images'):
			split0 = tf.slice(UNet2D.nn,[0,0,0,0],[-1,-1,-1,1])
			split1 = tf.slice(UNet2D.nn,[0,0,0,1],[-1,-1,-1,1])
			if nClasses > 2:
				split2 = tf.slice(UNet2D.nn,[0,0,0,2],[-1,-1,-1,1])
			tf.summary.image('pm0',split0)
			tf.summary.image('pm1',split1)
			if nClasses > 2:
				tf.summary.image('pm2',split2)
		merged = tf.summary.merge_all()


		# --------------------------------------------------
		# session
		# --------------------------------------------------

		saver = tf.train.Saver()
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

		if os.path.exists(outLogPath):
			shutil.rmtree(outLogPath)
		trainWriter = tf.summary.FileWriter(trainWriterPath, sess.graph)
		validWriter = tf.summary.FileWriter(validWriterPath, sess.graph)

		if restoreVariables:
			saver.restore(sess, outModelPath)
			print("Model restored.")
		else:
			sess.run(tf.global_variables_initializer())

		# --------------------------------------------------
		# train
		# --------------------------------------------------

		batchData = np.zeros((batchSize,imSize,imSize,nChannels))
		batchLabels = np.zeros((batchSize,imSize,imSize,nClasses))
		for i in range(nSteps):
			# train

			perm = np.arange(nTrain)
			np.random.shuffle(perm)

			for j in range(batchSize):
				batchData[j,:,:,:] = Train[perm[j],:,:,:]
				batchLabels[j,:,:,:] = LTrain[perm[j],:,:,:]

			summary,_ = sess.run([merged,optOp],feed_dict={UNet2D.tfData: batchData, tfLabels: batchLabels, UNet2D.tfTraining: 1})
			trainWriter.add_summary(summary, i)

			# validation

			perm = np.arange(nValid)
			np.random.shuffle(perm)

			for j in range(batchSize):
				batchData[j,:,:,:] = Valid[perm[j],:,:,:]
				batchLabels[j,:,:,:] = LValid[perm[j],:,:,:]

			summary, es = sess.run([merged, errors],feed_dict={UNet2D.tfData: batchData, tfLabels: batchLabels, UNet2D.tfTraining: 0})
			validWriter.add_summary(summary, i)

			e = np.mean(es)
			print('step %05d, e: %f' % (i,e))

			if i == 0:
				if restoreVariables:
					lowestError = e
				else:
					lowestError = np.inf

			if np.mod(i,100) == 0 and e < lowestError:
				lowestError = e
				print("Model saved in file: %s" % saver.save(sess, outModelPath))


		# --------------------------------------------------
		# test
		# --------------------------------------------------

		if not os.path.exists(outPMPath):
			os.makedirs(outPMPath)

		for i in range(nTest):
			j = np.mod(i,batchSize)

			batchData[j,:,:,:] = Test[i,:,:,:]
			batchLabels[j,:,:,:] = LTest[i,:,:,:]
		 
			if j == batchSize-1 or i == nTest-1:

				output = sess.run(UNet2D.nn,feed_dict={UNet2D.tfData: batchData, tfLabels: batchLabels, UNet2D.tfTraining: 0})

				for k in range(j+1):
					pm = output[k,:,:,testPMIndex]
					gt = batchLabels[k,:,:,testPMIndex]
					im = np.sqrt(normalize(batchData[k,:,:,0]))
					imwrite(np.uint8(255*np.concatenate((im,np.concatenate((pm,gt),axis=1)),axis=1)),'%s/I%05d.png' % (outPMPath,i-j+k+1))


		# --------------------------------------------------
		# save hyper-parameters, clean-up
		# --------------------------------------------------

		saveData(UNet2D.hp,pathjoin(modelPath,'hp.data'))

		trainWriter.close()
		validWriter.close()
		sess.close()

	def deploy(imPath,nImages,modelPath,pmPath,gpuIndex,pmIndex):
		os.environ['CUDA_VISIBLE_DEVICES']= '%d' % gpuIndex
		variablesPath = pathjoin(modelPath,'model.ckpt')
		outPMPath = pmPath

		hp = loadData(pathjoin(modelPath,'hp.data'))
		UNet2D.setupWithHP(hp)
		
		batchSize = UNet2D.hp['batchSize']
		imSize = UNet2D.hp['imSize']
		nChannels = UNet2D.hp['nChannels']
		nClasses = UNet2D.hp['nClasses']

		# --------------------------------------------------
		# data
		# --------------------------------------------------

		Data = np.zeros((nImages,imSize,imSize,nChannels))

		datasetMean = loadData(pathjoin(modelPath,'datasetMean.data'))
		datasetStDev = loadData(pathjoin(modelPath,'datasetStDev.data'))

		for iSample in range(0, nImages):
			path = '%s/I%05d_Img.tif' % (imPath,iSample)
			im = im2double(tifread(path))
			Data[iSample,:,:,0] = (im-datasetMean)/datasetStDev

		# --------------------------------------------------
		# session
		# --------------------------------------------------

		saver = tf.train.Saver()
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

		saver.restore(sess, variablesPath)
		print("Model restored.")

		# --------------------------------------------------
		# deploy
		# --------------------------------------------------

		batchData = np.zeros((batchSize,imSize,imSize,nChannels))

		if not os.path.exists(outPMPath):
			os.makedirs(outPMPath)

		for i in range(nImages):
			print(i,nImages)

			j = np.mod(i,batchSize)

			batchData[j,:,:,:] = Data[i,:,:,:]
		 
			if j == batchSize-1 or i == nImages-1:

				output = sess.run(UNet2D.nn,feed_dict={UNet2D.tfData: batchData, UNet2D.tfTraining: 0})

				for k in range(j+1):
					pm = output[k,:,:,pmIndex]
					im = np.sqrt(normalize(batchData[k,:,:,0]))
					# imwrite(np.uint8(255*np.concatenate((im,pm),axis=1)),'%s/I%05d.png' % (outPMPath,i-j+k+1))
					imwrite(np.uint8(255*im),'%s/I%05d_Im.png' % (outPMPath,i-j+k+1))
					imwrite(np.uint8(255*pm),'%s/I%05d_PM.png' % (outPMPath,i-j+k+1))


		# --------------------------------------------------
		# clean-up
		# --------------------------------------------------

		sess.close()

	def singleImageInferenceSetup(modelPath,gpuIndex):
		os.environ['CUDA_VISIBLE_DEVICES']= '%d' % gpuIndex
		variablesPath = pathjoin(modelPath,'model.ckpt')
		hp = loadData(pathjoin(modelPath,'hp.data'))
		UNet2D.setupWithHP(hp)

		UNet2D.DatasetMean =loadData(pathjoin(modelPath,'datasetMean.data'))
		UNet2D.DatasetStDev =  loadData(pathjoin(modelPath,'datasetStDev.data'))
		print(UNet2D.DatasetMean)
		print(UNet2D.DatasetStDev)

		# --------------------------------------------------
		# session
		# --------------------------------------------------

		saver = tf.train.Saver()
		UNet2D.Session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU
		#UNet2D.Session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
		saver.restore(UNet2D.Session, variablesPath)
		print("Model restored.")

	def singleImageInferenceCleanup():
		UNet2D.Session.close()

	def singleImageInference(image,mode,pmIndex):
		print('Inference...')

		batchSize = UNet2D.hp['batchSize']
		imSize = UNet2D.hp['imSize']
		nChannels = UNet2D.hp['nChannels']

		PI2D.setup(image,imSize,int(imSize/8),mode)
		PI2D.createOutput(nChannels)

		batchData = np.zeros((batchSize,imSize,imSize,nChannels))
		for i in range(PI2D.NumPatches):
			j = np.mod(i,batchSize)
			batchData[j,:,:,0] = (PI2D.getPatch(i)-UNet2D.DatasetMean)/UNet2D.DatasetStDev
			if j == batchSize-1 or i == PI2D.NumPatches-1:
				output = UNet2D.Session.run(UNet2D.nn,feed_dict={UNet2D.tfData: batchData, UNet2D.tfTraining: 0})
				for k in range(j+1):
					pm = output[k,:,:,pmIndex]
					PI2D.patchOutput(i-j+k,pm)
					# PI2D.patchOutput(i-j+k,normalize(imgradmag(PI2D.getPatch(i-j+k),1)))

		return PI2D.getValidOutput()


def identifyNumChan(path):
   tiff = tifffile.TiffFile(path)
   shape = tiff.pages[0].shape
   for i, page in enumerate(tiff.pages):
	   if page.shape != shape:
		   numChan = i
		   return numChan
		   break
#	   else:
#		   raise Exception("Did not find any pyramid subresolutions") 


def getProbMaps(I,dsFactor,modelPath):
   hsize = int((float(I.shape[0]) * float(0.5)))
   vsize = int((float(I.shape[1]) * float(0.5)))
   imagesub = cv2.resize(I,(vsize,hsize),cv2.INTER_NEAREST)

   UNet2D.singleImageInferenceSetup(modelPath, 0)

   for iSize in range(dsFactor):
	   hsize = int((float(I.shape[0]) * float(0.5)))
	   vsize = int((float(I.shape[1]) * float(0.5)))
	   I = cv2.resize(I,(vsize,hsize),cv2.INTER_NEAREST)
   I = im2double(I)
   I = im2double(sk.rescale_intensity(I, in_range=(np.min(I), np.max(I)), out_range=(0, 0.983)))
   probMaps = UNet2D.singleImageInference(I,'accumulate',1)
   UNet2D.singleImageInferenceCleanup()
   return probMaps 

def coreSegmenterOutput(I,initialmask,findCenter):
	hsize = int((float(I.shape[0]) * float(0.1)))
	vsize = int((float(I.shape[1]) * float(0.1)))
	nucGF = cv2.resize(I,(vsize,hsize),cv2.INTER_CUBIC)
	#active contours
	hsize = int(float(nucGF.shape[0]))
	vsize = int(float(nucGF.shape[1]))
	initialmask = cv2.resize(initialmask,(vsize,hsize),cv2.INTER_NEAREST)
	initialmask = dilation(initialmask,disk(15)) >0

	nucGF = gaussian(nucGF,0.7)
	nucGF=nucGF/np.amax(nucGF)
	
	nuclearMask = morphological_chan_vese(nucGF, 100, init_level_set=initialmask, smoothing=10,lambda1=1.001, lambda2=1)
	
	TMAmask = nuclearMask
	TMAmask = remove_small_objects(TMAmask>0,round(TMAmask.shape[0])*round(TMAmask.shape[1])*0.005)
	TMAlabel = label(TMAmask)
# find object closest to center
	if findCenter==True:
		
		stats= regionprops(TMAlabel)
		counter=1
		minDistance =-1
		index =[]
		for props in stats:
			centroid = props.centroid
			distanceFromCenter = np.sqrt((centroid[0]-nucGF.shape[0]/2)**2+(centroid[1]-nucGF.shape[1]/2)**2)
	#		if distanceFromCenter<0.6/2*np.sqrt(TMAlabel.shape[0]*TMAlabel.shape[1]):
			if distanceFromCenter<minDistance or minDistance==-1 :
				minDistance =distanceFromCenter
				index = counter
			counter=counter+1
	#		dist = 0.6/2*np.sqrt(TMAlabel.shape[0]*TMAlabel.shape[1])
		TMAmask = morphology.binary_closing(TMAlabel==index,disk(3))

	return TMAmask

def overlayOutline(outline,img):
	img2 = img.copy()
	stacked_img = np.stack((img2,)*3, axis=-1)
	stacked_img[outline > 0] = [1, 0, 0]
	imshowpair(img2,stacked_img)

def imshowpair(A,B):
	plt.imshow(A,cmap='Purples')
	plt.imshow(B,cmap='Greens',alpha=0.5)
	plt.show()


if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument("--imagePath")
	parser.add_argument("--outputPath")
	parser.add_argument("--maskPath")
	parser.add_argument("--tissue", action='store_true')
	parser.add_argument("--downsampleFactor", type=int, default = 5)
	parser.add_argument("--channel",type = int, default = 0)
	parser.add_argument("--buffer",type = float, default = 2)
	parser.add_argument("--outputChan", type=int, nargs = '+', default=[-1])
	parser.add_argument("--sensitivity",type = float, default=0.3)
	parser.add_argument("--useGrid",action='store_true')
	parser.add_argument("--cluster",action='store_true')
	args = parser.parse_args()

	outputPath = args.outputPath
	imagePath =  args.imagePath
	sensitivity = args.sensitivity
	scriptPath = os.path.dirname(os.path.realpath(__file__))
	modelPath = os.path.join(scriptPath, 'model')
	maskOutputPath = os.path.join(outputPath, 'masks')

	
#	if not os.path.exists(outputPath):
#		os.makedirs(outputPath)
#	else:
#		shutil.rmtree(outputPath)
	if not os.path.exists(maskOutputPath):
		os.makedirs(maskOutputPath)
	print(
		'WARNING! IF USING FOR TISSUE SPLITTING, IT IS ADVISED TO SET --downsampleFactor TO HIGHER THAN DEFAULT OF 5')
	channel = args.channel
	dsFactor = 1/(2**args.downsampleFactor)
	I = skio.imread(imagePath, img_num=channel)
	imagesub = resize(I,(int((float(I.shape[0]) * dsFactor)),int((float(I.shape[1]) * dsFactor))))
	numChan = identifyNumChan(imagePath)
	
	outputChan = args.outputChan
	if len(outputChan)==1:
		if outputChan[0]==-1:
			outputChan = [0, numChan-1]
		else:
			outputChan.append(outputChan[0])
	classProbs = getProbMaps(I, args.downsampleFactor, modelPath)

	if not args.tissue:
		print('TMA mode selected')
		preMask = gaussian(np.uint8(classProbs*255),1)>0.8

		P = regionprops(label(preMask),cache=False)
		area = [ele.area for ele in P]
		if len(P) <3:
			medArea = np.median(area)
			maxArea = np.percentile(area,99)
		else:
			count=0
			labelpreMask = np.zeros(preMask.shape,dtype=np.uint32)
			for props in P:
					count += 1
					yi = props.coords[:, 0]
					xi = props.coords[:, 1]
					labelpreMask[yi, xi] = count
					P=regionprops(labelpreMask)
					area = [ele.area for ele in P]
			medArea =  np.median(area)
			maxArea = np.percentile(area,99)
		preMask = remove_small_objects(preMask,0.2*medArea)
		coreRad = round(np.sqrt(medArea/np.pi))
		estCoreDiam = round(np.sqrt(maxArea/np.pi)*1.2*args.buffer)

		#preprocessing
		fgFiltered = blob_log(preMask,coreRad*0.6,threshold=sensitivity)
		Imax = np.zeros(preMask.shape,dtype=np.uint8)
		for iSpot in range(fgFiltered.shape[0]):
			yi = np.uint32(round(fgFiltered[iSpot, 0]))
			xi = np.uint32(round(fgFiltered[iSpot, 1]))
			Imax[yi, xi] = 1
		Imax = Imax*preMask
		Idist = distance_transform_edt(1-Imax)
		markers = label(Imax)
		coreLabel  = watershed(Idist,markers,watershed_line=True,mask = preMask)
		P = regionprops(coreLabel)
		centroids = np.array([ele.centroid for ele in P]) / dsFactor
		np.savetxt(outputPath + os.path.sep + 'centroidsY-X.txt', np.asarray(centroids), fmt='%10.5f')
		numCores = len(centroids)
		print(str(numCores) + ' cores detected!')
		estCoreDiamX = np.ones(numCores) * estCoreDiam / dsFactor
		estCoreDiamY = np.ones(numCores) * estCoreDiam / dsFactor
	else:
		print('Tissue mode selected')
		imageblur = 5
		Iblur = gaussian(np.uint8(255*classProbs), imageblur)
		coreMask = binary_fill_holes(binary_closing(Iblur > threshold_otsu(Iblur), np.ones((imageblur*2,imageblur*2))))
		coreMask = remove_small_objects(coreMask, min_size=0.001 * coreMask.shape[0] * coreMask.shape[1])

		## watershed
		Idist = distance_transform_edt(coreMask)
		markers = peak_local_max(h_maxima(Idist,20),indices=False)
		markers = label(markers).astype(np.int8)
		coreLabel = watershed(-Idist, markers, watershed_line=True,mask = coreMask)

		P = regionprops(coreLabel)
		centroids = np.array([ele.centroid for ele in P]) / dsFactor
		np.savetxt(outputPath + os.path.sep + 'centroidsY-X.txt', np.asarray(centroids), fmt='%10.5f')
		numCores = len(centroids)
		print(str(numCores) + ' tissues detected!')
		estCoreDiamX = np.array([(ele.bbox[3]-ele.bbox[1])*1.1 for ele in P]) / dsFactor
		estCoreDiamY = np.array([(ele.bbox[2]-ele.bbox[0])*1.1 for ele in P]) / dsFactor

	if numCores ==0 & args.cluster:
		print('No cores detected. Try adjusting the downsample factor')
		sys.exit(255)

	singleMaskTMA = np.zeros(imagesub.shape)
	maskTMA = np.zeros(imagesub.shape)
	bbox = [None] * numCores
	imagesub = imagesub/np.percentile(imagesub,99.9)
	imagesub = (imagesub * 255).round().astype(np.uint8)
	imagesub = gray2rgb(imagesub)
	x=np.zeros(numCores)
	xLim=np.zeros(numCores)
	y=np.zeros(numCores)
	yLim=np.zeros(numCores)
	
# segmenting each core   	
	#######################
	for iCore in range(numCores):
		x[iCore] = centroids[iCore,1] - estCoreDiamX[iCore]/2
		xLim[iCore] = x[iCore]+estCoreDiamX[iCore]
		if xLim[iCore] > I.shape[1]:
			xLim[iCore] = I.shape[1]
		if x[iCore]<1:
			x[iCore]=1

		y[iCore] = centroids[iCore,0] - estCoreDiamY[iCore]/2
		yLim[iCore] = y[iCore] + estCoreDiamY[iCore]
		if yLim[iCore] > I.shape[0]:
			yLim[iCore] = I.shape[0]
		if y[iCore]<1:
			y[iCore]=1

		bbox[iCore] = [round(x[iCore]), round(y[iCore]), round(xLim[iCore]), round(yLim[iCore])]
		coreStack = np.zeros((outputChan[1]-outputChan[0]+1,np.int(round(yLim[iCore])-round(y[iCore])-1),np.int(round(xLim[iCore])-round(x[iCore])-1)),dtype='uint16')

		for iChan in range(outputChan[0],outputChan[1]+1):
			with pytiff.Tiff(imagePath, "r", encoding='utf-8') as handle:
				handle.set_page(iChan)
				coreStack[iChan,:,:] =handle[np.uint32(bbox[iCore][1]):np.uint32(bbox[iCore][3]-1), np.uint32(bbox[iCore][0]):np.uint32(bbox[iCore][2]-1)]

		skio.imsave(outputPath + os.path.sep + str(iCore+1)  + '.tif',np.uint16(coreStack),imagej=True,bigtiff=True)
		with pytiff.Tiff(imagePath, "r", encoding='utf-8') as handle:
			handle.set_page(args.channel)
			coreSlice= handle[np.uint32(bbox[iCore][1]):np.uint32(bbox[iCore][3]-1), np.uint32(bbox[iCore][0]):np.uint32(bbox[iCore][2]-1)]

		core = (coreLabel ==(iCore+1))
		initialmask = core[np.uint32(y[iCore] * dsFactor):np.uint32(yLim[iCore] * dsFactor),
					  np.uint32(x[iCore] * dsFactor):np.uint32(xLim[iCore] * dsFactor)]
		if not args.tissue:
			initialmask = resize(initialmask,size(coreSlice),cv2.INTER_NEAREST)

			singleProbMap = classProbs[np.uint32(y[iCore]*dsFactor):np.uint32(yLim[iCore]*dsFactor),np.uint32(x[iCore]*dsFactor):np.uint32(xLim[iCore]*dsFactor)]
			singleProbMap = resize(np.uint8(255*singleProbMap),size(coreSlice),cv2.INTER_NEAREST)
			TMAmask = coreSegmenterOutput(coreSlice,initialmask,False)
		else:
			Irs = resize(coreSlice,(int((float(coreSlice.shape[0]) * 0.25)),int((float(coreSlice.shape[1]) * 0.25))))
			TMAmask = coreSegmenterOutput(Irs, np.uint8(initialmask), False)

		if np.sum(TMAmask)==0:
			TMAmask = np.ones(TMAmask.shape)
		vsize = int(float(coreSlice.shape[0]))
		hsize = int(float(coreSlice.shape[1]))
		masksub = resize(resize(TMAmask,(vsize,hsize),cv2.INTER_NEAREST),(int((float(coreSlice.shape[0])*dsFactor)),int((float(coreSlice.shape[1])*dsFactor))),cv2.INTER_NEAREST)
		singleMaskTMA[int(y[iCore]*dsFactor):int(y[iCore]*dsFactor)+masksub.shape[0],int(x[iCore]*dsFactor):int(x[iCore]*dsFactor)+masksub.shape[1]]=masksub
		maskTMA = maskTMA + resize(singleMaskTMA,maskTMA.shape,cv2.INTER_NEAREST)

		cv2.putText(imagesub, str(iCore+1), (int(P[iCore].centroid[1]),int(P[iCore].centroid[0])), 0, 0.5, (0,255,0), 1, cv2.LINE_AA)
		
		skio.imsave(maskOutputPath + os.path.sep + str(iCore+1)  + '_mask.tif',np.uint8(TMAmask))
		print('Segmented core/tissue ' + str(iCore+1))
		
	boundaries = find_boundaries(maskTMA)
	imagesub[boundaries==1] = 255
	skio.imsave(outputPath + os.path.sep + 'TMA_MAP.tif' ,imagesub)
	print('Segmented all cores/tissues!')

#restore GPU to 0
	#image load using tifffile
