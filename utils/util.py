def get_torch_model():
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    return resnet18

def get_example_input(image_file):
    """
    Load image from disk and converts to compatible shape
    :param image_file: Path to single image file
    :return: Orginal image, numpy.ndarray instance image, torch.Tensor image
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(torch.device("cpu"))
    print(torch_img.shape)
    return image, torch_img.numpy(), torch_img