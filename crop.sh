echo "parse training cropping"
python preprocessing_data/image_cropping.py\
  -- source_folder './dataset/rfmid/Training_Set/Training'\
  -- destination_folder './dataset/data_preprocessing/training_set/training_cropped'\
  #-- cpus 4\
  
#echo "parse testing cropping"
#python preprocessing_data/image_cropping.py\
#  -- source_folder './dataset/rfmid/Test_Set/Testing'\
#  -- destination_folder './dataset/data_preprocessing/testing_set/testing_cropped'\
  #-- cpus 4\

echo "parse validation cropping"
python preprocessing_data/image_cropping.py\
  -- source_folder './dataset/rfmid/Evaluation_Set/Validation'\
  -- destination_folder './dataset/data_preprocessing/validdation_set/validation_cropped'\
  #-- cpus 4\
