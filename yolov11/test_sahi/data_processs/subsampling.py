from PIL import Image

input_image_path = '/media/c515/AAA82521A824ED8F/data/lmx/cotton/cotton4/test/images/6122901357544455895_jpg.rf.28082d32ebe990bc5c2a57bf10a73c72.jpg'  
output_image_path = '/home/c515/lmx/ultralytics-8.3.35/test_sahi/data_processs/data_processed/2.jpg'  

height_org = 480
width_org = 640

height_aim = 120
width_aim =160

height_back =480
width_back =640

original_image = Image.open(input_image_path)

resized_image = original_image.resize((width_aim, height_aim))

background = Image.new('RGB', (width_back, height_back), (255, 255, 255))  # 白色背景

left = (width_org - width_aim) // 2  
top = (height_org - height_aim) // 2  

background.paste(resized_image, (left, top))

background.save(output_image_path)
print(f"图像已保存到 {output_image_path}")
