class HyperParametersBase:
    def __init__(self, full_cases_count, feature_num, target_num):
        self.train_ratio = 0.95
        self.full_cases_count = full_cases_count
        self.feature_num = feature_num
        self.target_num = target_num
        self.num_epochs = 20
        self.batch_size = 128
        self.lr_decay_rate = 0.95
        self.initial_lr = 0.001

        self.train_size = 0
        self.validation_size = 0
        self.set_train_validation_size()

    def set_num_epochs(self, new_num_epochs):
        self.num_epochs = new_num_epochs

    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size

    def set_lr_decay_rate(self, new_lr_decay_rate):
        self.lr_decay_rate = new_lr_decay_rate

    def initial_lr(self, new_initial_lr):
        self.initial_lr = new_initial_lr

    def set_train_validation_size(self):
        self.train_size = int(self.full_cases_count * self.train_ratio)
        self.validation_size = int(
            self.full_cases_count * (1 - self.train_ratio))

    def set_train_ratio(self, new_train_ratio):
        self.train_ratio = new_train_ratio
        self.set_train_validation_size()
