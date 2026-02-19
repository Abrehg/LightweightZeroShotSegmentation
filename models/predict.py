import torch
from clip_model import CLIPTokenize, create_text_encoder
from prior_model import create_prior
from SAM_model import create_SAM
from distill_model import create_Student, DistilledMemoryStudent

class TeacherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = create_text_encoder()
        self.prior = create_prior()
        self.sam_decoder = create_SAM()

    def forward(self, images, text):
        text_tokens = CLIPTokenize(text)
        text_emb = self.text_encoder(text_tokens)
        prior_emb = self.prior(text_emb)
        result = self.sam_decoder(images, prior_emb)
        return result
    
    def load_weights(self, text_weights_file, prior_weights_file, decoder_weights_file):
        self.text_encoder.load_weights(text_weights_file)
        self.prior.load_weights(prior_weights_file)
        self.sam_decoder.load_weights(decoder_weights_file)

class StudentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.student:DistilledMemoryStudent = create_Student()
    
    def forward(self, images, text):
        text_tokens = CLIPTokenize(text)
        out = self.student(images, text_tokens)
        return out
    
    def load_weights(self, student_weights):
        self.student.load_weights(student_weights)

teacher = TeacherModel()