import torch
import torch.nn as nn
import torch.nn.functional as F


def ITC_loss(image_features, text_features, pid, temperature):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)


    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = temperature * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def cmpm_loss(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss


def tcmpm_loss(image_embeddings, text_embeddings, pid, image_id, temperature, label_mix=False, epsilon=1e-8):
    """
    Temperature-scaled Cross-Modal Projection Matching Loss(TCMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: tcmpm loss for image projected to text
        t2i_loss: tcmpm loss for text projected to image
    """

    batch_size = image_embeddings.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels_mask = (pid_dist == 0).float()

    if label_mix:
        image_id = image_id.reshape((batch_size, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels_mask = (labels_mask + image_id_mask) / 2

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    image_proj_text = temperature * torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = temperature * torch.matmul(text_embeddings, image_norm.t())
        
    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.sum(dim=1) # original paper use sum, and use norm will lead minus loss
    # labels_mask_norm = labels_mask / labels_mask.norm(dim=1)
    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    tcmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return tcmpm_loss