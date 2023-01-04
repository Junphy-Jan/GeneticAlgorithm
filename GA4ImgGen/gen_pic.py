from typing import Union, List

import ray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided


def pool2d(input_m, kernel_size, stride, padding, pool_mode='max'):
    """
    2D Pooling

    Parameters:
        input_m: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    input_m = np.pad(input_m, padding, mode='constant')

    # Window view of A
    output_shape = ((input_m.shape[0] - kernel_size) // stride + 1,
                    (input_m.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(input_m, shape=output_shape + kernel_size,
                     strides=(stride * input_m.strides[0],
                              stride * input_m.strides[1]) + input_m.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def show_img(img_data, title="no-title", gray=False, save=False):
    """
    画图
    :param save:
    :param img_data: (width, height, optional(channel))
    :param title: 图像名
    :param gray: 是否灰度图
    :return:
    """
    plt.title(title)
    if gray:
        plt.imshow(img_data, cmap="gray")
    else:
        plt.imshow(img_data)
    if save:
        plt.savefig("../data/generated_{}.jpg".format(title))
    plt.show()


def get_target_pic(pic_path):
    img = Image.open(pic_path)  # 读入图片数据
    images = []
    if img.mode.__contains__("RGB"):
        show_img(img, "original pic")
        img = np.array(img)  # 转换为numpy
        img_width, img_height, c = img.shape
        img_r = img[:, :, 0]
        # show_img(img_r, "channel r", gray=True)
        img_g = img[:, :, 1]
        # show_img(img_g, "channel g")
        img_b = img[:, :, 2]
        # show_img(img_b, "channel b")
        # show_img(np.stack((img_r, img_g, img_b), axis=2), "composed")
        images.append(img_r)
        images.append(img_g)
        images.append(img_b)
        return images, img_width, img_height
    elif img.mode == "L":
        show_img(img, "original pic", gray=True)
        img = np.array(img)  # 转换为numpy
        img_width, img_height = img.shape
        images.append(img)
        return images, img_width, img_height


# 初始化种群
def init_pops(amount, img_width, img_height):
    pops = []
    for i in range(amount):
        pops.append(np.random.randint(0, 256, size=(img_width * img_height,), dtype=int))
    show_img(pops[0].reshape(img_width, img_height), title="original pop pic")
    return np.array(pops)


def calc_loss(pops, targets: Union[List, np.ndarray], specified=0):
    """
    计算单个个体与目标的差距
    :param pops: N * (784,)
    :param targets: (784,)
    :param specified: 指定目标
    :return:
    """
    if isinstance(targets, List):
        loss = []
        for pop in pops:
            target = targets[specified]
            abs_dist = np.maximum(pop - target, target - pop)
            loss.append(np.sum(abs_dist) / len(target))
        return np.array(loss)
    else:
        abs_dist = np.maximum(pops - targets, targets - pops)
        loss = np.sum(abs_dist, axis=1) / len(targets)
        return loss


def cal_pooled_loss(pops, width, height, targets: Union[List, np.ndarray], specified=0, mode="avg"):
    pooled_pops = []
    for pop in pops:
        pooled_pop = pool2d(pop.reshape(width, height), kernel_size=2, stride=2, padding=0, pool_mode=mode)
        pooled_pops.append(pooled_pop.reshape((-1,)))
    target = pool2d(targets[specified].reshape(width, height), kernel_size=2, stride=2, padding=0,
                    pool_mode=mode).reshape((-1,))
    return calc_loss(np.array(pooled_pops), target)


def get_fitness(loss):
    """
    计算每个个体的适应度
    :param loss: 每个个体与目标的差距集合
    :return:
    """
    return -(loss - np.max(loss)) + 1e-3


def select(pops, fitness, pop_size, choose_one=False):  # nature selection wrt pop's fitness
    if choose_one:
        idx = np.argmax(fitness)
    else:
        idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True,
                               p=fitness / (fitness.sum()))
    # print("选择的个数：{}".format(len(np.unique(idx))))
    return pops[idx]


def mutation(child, dna_size, mutation_rate=0.05):
    if np.random.rand() < mutation_rate:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, dna_size)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = np.random.randint(0, 256)  # 随机产生一个0-255的整数
    return child


def crossover_and_mutation(pops, pop_size, dna_size, crossover_rate=0.8, mutation_rate=0.05):
    new_pops = []
    for father in pops:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因
        if np.random.rand() < crossover_rate:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pops[np.random.randint(pop_size)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=dna_size)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        child = mutation(child, dna_size, mutation_rate)  # 每个后代有一定的机率发生变异
        new_pops.append(child)
    return np.array(new_pops)


def split_img(img, width, height, max_width, max_height):
    """
    将图片分为N份
    :param img: ndarray(width, height)
    :param width:
    :param height:
    :param max_width:
    :param max_height:
    :return:
    """
    while height % max_height != 0:
        max_height -= 1
    while width % max_width != 0:
        max_width -= 1
    width_piece = int(width / max_width)
    height_piece = int(height / max_height)
    print("分割的图片的宽度：{}，高度：{}，份数：{}".format(max_width, max_height, width_piece * height_piece))
    split_images = []
    for i in range(height_piece):
        for j in range(width_piece):
            split_images.append(img[i * max_height: (i + 1) * max_height, j * max_width: (j + 1) * max_width])
    return split_images, max_width, max_height


@ray.remote
def generate_pic(target, dna_size, pop_size, generations, img_width, img_height,
                 crossover_rate=0.8, max_mutation_rate=0.3, min_mutation_rate=0.03,
                 print_times=10):
    pops_ = init_pops(pop_size, img_width, img_height)
    print_n = int(generations / print_times)
    mutation_rate = np.linspace(max_mutation_rate, min_mutation_rate, generations)
    fitness_ = np.empty((pop_size,))
    specified = 0
    changed_gen = int(generations * 0.7)
    for generation in tqdm(range(generations)):
        pops_ = crossover_and_mutation(pops_, pop_size, dna_size, crossover_rate, mutation_rate[generation])
        if generation >= changed_gen:
            # print("拟合目标换成：{}".format(specified))
            specified = 1
        loss_ = calc_loss(pops_, target, specified=specified)
        fitness_ = get_fitness(loss_)
        # print("适应度：{}".format(fitness_))
        pops_ = select(pops_, fitness_, pop_size)
        if (generation + 1) % print_n == 0:
            # 打印以下种群中第一幅图片
            show_img(pops_[0].reshape(img_width, img_height), title="pic of generation:{}".format(generation + 1))
            # print("与第：{}/{} 个目标的差距：{}".format(specified, len(target), loss_))
    best_pop = select(pops_, fitness_, pop_size, choose_one=True)
    show_img(best_pop.reshape(img_width, img_height), title="best pic")
    return best_pop.reshape(img_width, img_height)


def run_instance():
    pic_path_1 = "../data/demo1.jpg"
    target_pic_1, img_width_, img_height_ = get_target_pic(pic_path_1)
    print("图像宽度：{}，高度：{}".format(img_width_, img_height_))
    DNA_SIZE = img_width_ * img_height_
    POP_SIZE = 40
    GENERATIONS = 30000

    show_img(target_pic_1[0], title="target pic")
    split_imgs, split_width, split_height = split_img(target_pic_1[0], img_width_, img_height_, max_width=80,
                                                      max_height=80)
    split_best = []
    for img in split_imgs:
        show_img(img, title="split target pic")
        DNA_SIZE = img.shape[1] * img.shape[0]
        print("dna size:{}".format(DNA_SIZE))
        best_one = generate_pic.remote(
            img.reshape((-1,)), DNA_SIZE, pop_size=POP_SIZE,
            generations=GENERATIONS, img_width=split_width, img_height=split_height,
            crossover_rate=0.8, max_mutation_rate=0.5, min_mutation_rate=0.05,
            print_times=10
        )
        split_best.append(best_one)

    split_best = ray.get(split_best)
    print(len(split_best))

    width_piece = int(img_width_ / split_width)
    height_piece = int(img_height_ / split_height)
    height_pieces = []
    for i in range(width_piece):
        height_pieces.append(np.concatenate(split_best[i * width_piece: (i + 1) * width_piece], axis=1))
    best_img = np.concatenate(height_pieces)
    show_img(best_img, title="composed best img", save=True)


if __name__ == '__main__':
    run_instance()
