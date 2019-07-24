# PSS

tobacco_download_unpack.py: downloads .cpio folders from RVL_CDIP, unzips them and splits every image (if it is multipage) that match with BigTobacco subset. Delete every spare image and downloaded .cpio folder. Writes csv file with new image directory, class (number) and page segmentation.

SmallTobacco_download_unpack.py: downloads .cpio folders from RVL_CDIP, unzips them and splits every image (if it is multipage) that match with SmallTobacco subset. Writes csv file with new image directory, class (string) and page segmentation.

3482_comprobation.py: takes SmallTobacco.csv file output from SmallTobacco_download_unpack.py and changes string labels by number (0-9). Can also be used to check if every image in SmallTobacco (SmallTobacco_files.csv) is really in csv (SmallTobacco.csv) with split SmallTobacco info. 3492 should be result obtained.

ocr_tobacco.py: performs tesseract OCR to any csv file with image directories, creates txt file with OCR text in the same path as the image and stores the txt path in another csv file (copies initial csv info too).

SmallTobacco_csv.py: creates csv file with image and class from a SmallTobacco joint foulder with individual images, as downloaded from https://lampsrv02.umiacs.umd.edu/projdb/project.php?id=72

hdf5_test.py: reads inputs csv with images directories, class (and page segmentation to be added) and creates train, validation and test hdf5 datasets (select compression method). For images datasets, open image and stores it (cv2 library).

HDF5_pytorch_SmallTobacco.py: implementation of dataloader which reads hdf5 file and CNN Text + MobilenetNet concatenation

fasttext_corpus.py: generates FastText format for sentence classification

fusion_pretrained_v2.py: Image + Text NN model
