"""
PyRadiomics Feature Extraction Module
Extracts radiomics features from CT images using PyRadiomics library.
"""

from __future__ import print_function
import logging
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import radiomics

radiomics.logger.setLevel(logging.ERROR)

def Numpy2Itk(array):
    """
    Convert numpy array to SimpleITK image format.
    
    Args:
        array: numpy array
    
    Returns:
        SimpleITK image
    """
    return sitk.GetImageFromArray(array)

def radiomics_extract(image_origin, image_mask, 
                     features=['firstorder', 'shape', 'glcm', 'glszm', 'glrlm', 'ngtdm', 'gldm'],
                     imagetypes=['Original', 'LoG', 'Wavelet']):
    """
    Extract radiomics features from image and mask.
    
    Args:
        image_origin: Image array (numpy array)
        image_mask: Mask array (numpy array)
        features: List of feature classes to extract
        imagetypes: List of image types to process
    
    Returns:
        feats: List of feature values
        cols: List of feature names
    """
    image = Numpy2Itk(image_origin)
    mask = Numpy2Itk(image_mask)

    settings = {}
    settings['binWidth'] = 2.8
    settings['interpolator'] = 'sitkGaussian'
    settings['resampledPixelSpacing'] = [0, 0, 0]
    settings['force2D'] = True
    settings['verbose'] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.settings['enableCExtensions'] = True
    
    for feature in features:
        extractor.enableFeatureClassByName(feature.lower())

    for imagetype in imagetypes:
        extractor.enableImageTypeByName(imagetype, enabled=True, customArgs=None)
        
    featureVector = extractor.execute(image, mask)
    
    cols = []
    feats = []
    for feature in features:
        for featureName in sorted(featureVector.keys()):
            if feature in featureName:
                cols.append(featureName)
                feats.append(featureVector[featureName])
                
    return feats, cols
