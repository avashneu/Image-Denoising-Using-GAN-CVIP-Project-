1. Steps for Building Data Pipeline
  a. Split the data into training and test set
  b. For each training and test set, perform the following operations:
    - Resize the image ; image_height, image_width = 128,128
    - Grayscale the image
    - Convert the image to tesorflow array
    - Normalize the array: (i.e just divide the tensorflow array by 255.0)
  
  c. Create a batch of training and test set
  
  Note: Generator takes in NOISY IMAGES
        Discriminator takes the output from generator to give Dg
        Discriminator also take the ground truth image to give Dx
        These values Dg and Dx will be used in our loss function.

2. Using this batch of images, we'll call our GAN model
  
