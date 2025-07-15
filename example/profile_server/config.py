class Config:
    def __init__(self, model_config=None, sampling_config=None, server_config=None, prompt_config=None, profile_config=None, served_model_name=None):

        self.model_config = model_config
        self.sampling_config = sampling_config
        self.server_config = server_config
        self.prompt_config = prompt_config
        self.profile_config = profile_config
        self.served_model_name = served_model_name