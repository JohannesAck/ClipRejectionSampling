import numpy as np
import torch
import matplotlib.pyplot as plt
import clip
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
from generators import generate_styleganv2

device = "cpu"


def rejection_sample(probs: torch.Tensor, target_dist: np.ndarray):
    """
    Rejection sample based on previously calculated probabilities.
    """
    assert probs.shape[1:] == target_dist.shape
    target_dist = torch.Tensor(target_dist).to(probs.device)

    eps = 1e-6
    ratio_upper_bound = torch.max(
        torch.tile(target_dist, [probs.shape[0], 1])/ (probs + eps)
    )
    uniform = torch.rand(probs.shape[0])
    likelihood_ratios = target_dist / ((probs + eps) * ratio_upper_bound)
    accepted = uniform < likelihood_ratios
    accepted_idxs = torch.nonzero(accepted)
    return accepted_idxs


def main():
    model, preprocess = clip.load("ViT-B/32", device=device, download_root='./data/clip')
    # model, preprocess = clip.load("ViT-L/14", device=device, download_root='./data/clip')
    tokenize_fun = clip.tokenize

    positive_prompt = 'woman'
    negative_prompt = 'man'
    n_classes = 2
    with torch.no_grad():
        text_tokens = tokenize_fun([positive_prompt, negative_prompt]).to(device)
        text_enc = model.encode_text(text_tokens).float()

    # pil_imgs = np.load('dalle_mini_an_asian_person_32.npy', allow_pickle=True).tolist()
    n_sample = 64
    imgs_tensor, imgs_pil = generate_styleganv2('./data/stylegan2-ffhq-256x256.pkl', n_sample, batch_size=32)

    print('convert to tensor')
    processed = []
    for img in imgs_pil:
        processed.append(preprocess(img))
    data = torch.stack(processed)
    print('convert do clip classification')
    image_features = model.encode_image(data)
    cos_sim = image_features @ text_enc.T
    print(cos_sim)
    probs = torch.softmax(100.0 * cos_sim, 1).detach()

    accepted_idxs = rejection_sample(probs, np.full([n_classes], 1.0 / n_classes))

    for idx in range(len(imgs_pil[:100])):
        plt.imshow(np.array(imgs_pil[idx]))
        plt.title(f'woman: {probs[idx][0]:.2f}, man: {probs[idx][1]:.2f}')
        plt.savefig(f'out/fig{idx}.png')
        plt.close()


if __name__ == '__main__':
    main()
