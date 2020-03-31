import torch


class GeneratedDataset(torch.utils.data.IterableDataset):

    def __init__(self, length, device=torch.device(0), generate_online=True):
        super().__init__()
        self.__seed = 2
        self.__length = length
        self.__device = device
        self.__generate_online = generate_online
        if device.type == "cuda":
            self.__generator = torch.cuda.manual_seed(self.__seed)  # returns None, does not work properly with CUDA
        else:
            self.__generator = torch.manual_seed(self.__seed)

        if not self.__generate_online:
            self.__data = torch.randint(0, 2, (self.__length, 2), generator=self.__generator, device=self.__device)

    def _function(self, a, b):
        raise NotImplementedError("Dataset output value function generator not implemented!")

    def __get_data(self, gen_length):
        if self.__generate_online:
            return torch.randint(0, 2, (gen_length, 2), generator=self.__generator, device=self.__device)
        else:
            return self.__data

    def __generate(self, gen_length):
        data = self.__get_data(gen_length)
        for x in data.unbind(dim=0):
            x1, x2 = x.unbind()
            y = self._function(x1, x2)
            yield(x, y)

    def __len__(self):
        return self.__length

    def __iter__(self):
        return self.__generate(self.__length)


class ORSet(GeneratedDataset):

    def _function(self, a, b):
        return a | b


class ANDSet(GeneratedDataset):

    def _function(self, a, b):
        return a & b


class XORSet(GeneratedDataset):

    def _function(self, a, b):
        return a ^ b
