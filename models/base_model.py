import torch


class BaseModel(torch.nn.Module):
    """模型基类（实现公共方法）"""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.__dict__
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
