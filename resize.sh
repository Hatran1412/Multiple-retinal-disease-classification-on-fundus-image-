echo "parse training resizing"
python preprocessing_data/image_resizing.py\
  -- source_folder './dataset/data_preprocessing/training_set/training_cropped'\
  -- destination_folder './dataset/data_preprocessing/training_set/training_resized'\
  #-- cpus 4\
  
echo "parse testing resizing"
python preprocessing_data/image_resizing.py\
  -- source_folder './dataset/data_preprocessing/testing_set/testing_cropped'\
  -- destination_folder './dataset/data_preprocessing/testing_set/testing_resized'\
  #-- cpus 4\

echo "parse validation resizing"
python preprocessing_data/image_resizing.py\
  -- source_folder './dataset/data_preprocessing/validation_set/validation_cropped'\
  -- destination_folder './dataset/data_preprocessing/validation_set/validation_resized'\
  #-- cpus 4\