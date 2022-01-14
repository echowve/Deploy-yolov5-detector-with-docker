from flask_restful import reqparse
import werkzeug

class Parser():

    def __init__(self):

        self.parser = reqparse.RequestParser()
        self.parser.add_argument("task", type=str, default='binary')
        self.parser.add_argument("data_type", type=int, default=1)
        self.parser.add_argument("data_file", type=str)
        self.parser.add_argument("model_type", type=str, default="keras", choices=["keras", "pytorch"])
        self.parser.add_argument("model_path", type=str)
        self.parser.add_argument("input_size", type=int, default=(224, 224))
        self.parser.add_argument("sub_path", type=str, default=None)
        self.parser.add_argument("img_path", type=str, default=None)
        self.parser.add_argument("shapex", type=int, default=None)
        self.parser.add_argument("shapey", type=int, default=None)
        # self.parser.add_argument("files", type=bytes, default=None)
        self.parser.add_argument("files", type=werkzeug.datastructures.FileStorage, location='files',default=None)
    def getparser(self):
        return self.parser
 

