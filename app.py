# 框架包
from flask import Flask ,request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS,cross_origin
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from flask_migrate import Migrate
from flask_uuid import FlaskUUID
import uuid
from datetime import datetime
basedir = os.path.abspath(os.path.dirname(__file__))
# 框架包

# 预测用包
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import matplotlib
from torchvision import transforms
from PIL import Image
# 预测用包

# 预测配置
pre_base_dir = './static/resources/'
device = torch.device('cpu')
matplotlib.rc("font",family='SimHei')
font = ImageFont.truetype(pre_base_dir+'SimHei.ttf', 32)
idx_to_labels = np.load(pre_base_dir+'idx_to_labels.npy', allow_pickle=True).item()
model = torch.load('./static/model/best-0.904.pth')
model = model.eval().to(device)
# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])


# 框架配置
app = Flask(__name__)
# 使用uuid生成id
FlaskUUID(app)

HOSTNAME = "127.0.0.1"
PORT = 3306
USERNAME = 'root'
PASSWORD = '13086186924'
DATABASE = 'S'
app.config['SQLALCHEMY_DATABASE_URI']=f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8"

cors = CORS(app, resources={r"/*": {"origins": "*"}})  #允许所有跨域


db = SQLAlchemy(app)
migrate = Migrate(app, db)

# 测试数据库链接
# with app.app_context():
#     with db.engine.connect() as con:
#         res = con.execute("select 1")
#         print(res.fetchone())

class UserInfo(db.Model):
    __tablename__ = 'user_table'
    user_id = db.Column('user_id', db.String(50), primary_key=True) #用户id
    user_name = db.Column('user_name', db.String(50), unique=True) # 用户名
    user_psw = db.Column('user_psw', db.String(50)) #用户密码
    user_phone = db.Column('user_phone', db.String(11)) #用户电话
    user_sex = db.Column('user_sex', db.String(2)) #用户性别
    user_portrait_file = db.Column('user_portrait_file', db.String(50), default='') #用户头像文件名
    user_type = db.Column('user_type', db.Integer, default=0) #是否为管理员
    files = db.relationship('FileInfo', back_populates='user') #与file表中的user建立关系

    def keys(self):
        return ['user_type', 'user_id','user_name', 'user_psw', 'user_phone', 'user_sex', "user_portrait_file"]

    def __getitem__(self, item):
        return self.__getattribute__(item)


class FileInfo(db.Model):
    __tablename__ = 'file_table'
    file_name = db.Column('file_name', db.String(50)) #文件名
    file_id = db.Column('file_id', db.String(50), primary_key=True) #文件id
    user_id = db.Column('user_id', db.String(50), db.ForeignKey("user_table.user_id")) #文件上传者
    res_name = db.Column('res_name', db.String(50)) #测试结果文件名
    class_name = db.Column('class_name', db.String(20))
    add_time = db.Column('add_time', db.DateTime, default=datetime.now().astimezone())
    user = db.relationship('UserInfo', back_populates="files") #与外键反向引用

    def keys(self):
        return ['file_name', 'file_id', 'user_id', 'res_name', 'class_name', 'add_time']

    def __getitem__(self, item):
        return self.__getattribute__(item)

@app.route('/')
def hello_world():
    return 'ok'

@app.route('/process_img',methods=('GET', 'POST'))
@cross_origin()
def process_img():
    # 保存图片
    f = request.files.get('file')
    # print(user_id)
    # return 'ok'
    filename = secure_filename(f.filename)
    print(filename)
    filename = datetime.now().strftime("%Y%m%d%H%M%S")+'_.'+filename.rsplit('.', 1)[1]
    file_path = basedir + "/static/file/" + filename
    f.save(file_path)

    # 处理图片
    img_pil = Image.open(file_path)
    input_img = test_transform(img_pil)
    input_img = input_img.unsqueeze(0).to(device)
    pred_logits = model(input_img)
    pred_softmax = F.softmax(pred_logits, dim=1)

    plt.figure(figsize=(22, 10))
    x = idx_to_labels.values()
    y = pred_softmax.cpu().detach().numpy()[0] * 100
    width = 0.45  # 柱状图宽度
    ax = plt.bar(x, y, width)

    n = 10
    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析出置信度

    # 画图
    draw = ImageDraw.Draw(img_pil)
    for i in range(n):
        class_name = idx_to_labels[pred_ids[i]]  # 获取类别名称
        confidence = confs[i] * 100  # 获取置信度
        text = '{:<15} {:>.4f}'.format(class_name, confidence)
        # 文字坐标，中文字符串，字体，rgba颜色
        draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))
    fig = plt.figure(figsize=(18, 6))

    # 绘制左图-预测图
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_pil)
    ax1.axis('off')

    # 绘制右图-柱状图
    ax2 = plt.subplot(1, 2, 2)
    x = idx_to_labels.values()
    y = pred_softmax.cpu().detach().numpy()[0] * 100
    ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
    plt.bar_label(ax, fmt='%.2f', fontsize=10)  # 置信度数值

    plt.title('{} 图像分类预测结果'.format(filename), fontsize=30)
    plt.xlabel('类别', fontsize=20)
    plt.ylabel('置信度', fontsize=20)
    plt.ylim([0, 110])  # y轴取值范围
    ax2.tick_params(labelsize=16)  # 坐标文字大小
    plt.xticks(rotation=90)  # 横轴文字旋转
    plt.tight_layout()
    fig.savefig('./static/outputs/'+filename)

    # 将文件信息添加到数据库
    possible_class_name = idx_to_labels[pred_ids[0]]
    user_id = request.form['user_id']
    file_id = uuid.uuid4()
    new_file = FileInfo(file_id=file_id, user_id=user_id, file_name=filename, res_name=filename, class_name=possible_class_name)
    db.session.add(new_file)
    db.session.commit()

    return {'msg':'success', 'code':0, 'data':{'filename':filename}}



@app.route('/login', methods=('GET', 'POST'))
def login():
    try:
        temp = []
        data = eval(str(request.data, encoding="utf-8"))
        user = UserInfo.query.filter(UserInfo.user_name == data['user_name'] and UserInfo.user_psw == data['user_psw']).first()
        temp.append(user)
        if len(temp) == 1:
            return {'msg':'ok', 'code':0, 'data':dict(user)}
        else:
            return {'msg':'invalid information', 'code':1}
    except Exception as r:
        print('%s' %r)
        return{'msg':'err','code':1}


@app.route('/register', methods=('GET', 'POST'))
def addUser():
    try:
        random_uuid = uuid.uuid4()
        print(random_uuid)
        user_info = eval(str(request.data, encoding='utf-8'))
        new_user = UserInfo(user_id=random_uuid, user_name=user_info['user_name'], user_psw=user_info['user_psw'], user_phone=user_info['user_phone'], user_sex=user_info['user_sex'])
        db.session.add(new_user)
        db.session.commit()
        return {'msg':'ok', 'data':dict(new_user),'code':0}
    except Exception as r:
        print('%s'%r)
        return {'msg':'err', 'code':1}


@app.route('/get_files', methods=('GET', 'POST'))
def getFiles():
    user_id = eval(str(request.data, encoding='utf-8'))['user_id']
    user = db.session.query(UserInfo).get(user_id)
    files = user.files
    files = [dict(file) for file in files]
    return files

@app.route('/get_file_num', methods=('GET', 'POST'))
def getFileNum():
    user_id = request.args['user_id']
    user = db.session.query(UserInfo).get(user_id)
    nums = len([file for file in user.files])
    return str(nums)

@app.route('/get_users')
def getUsers():
    try:
        users = db.session.query(UserInfo).all()
        users = [dict(user) for user in users]
        return users
    except Exception as e:
        print('%s' % e)
        return 'err'

@app.route('/test')
def test():
    user = UserInfo.query.get(1)
    files = user.files
    data = [dict(file) for file in files]
    # data = []
    # n=0
    # for file in files:
    #     temp = {}
    #     temp['file_name'] = file.file_name
    #     temp['file_id'] = file.file_id
    #     temp['res_name'] = file.res_name
    #     data.append(temp)
    return data

@app.route('/delete_record',methods=['POST','GET'])
def delete_record():
    file_id = eval(str(request.data, encoding='utf-8'))['file_id']
    file = FileInfo.query.get(file_id)
    db.session.delete(file)
    db.session.commit()
    return {'msg':'删除成功'}


@app.route('/update_user_info',methods=['POST','GET'])
def update_user_info():
    user_info = eval(str(request.data, encoding='utf-8'))
    user = UserInfo.query.filter_by(user_id = user_info['user_id']).first()
    if user:
        user.user_name = user_info['user_name']
        user.user_sex = user_info['user_sex']
        user.user_phone = user_info['user_phone']
        user.user_psw = user_info['user_psw']
        db.session.commit()
        new_user = UserInfo.query.filter_by(user_id = user_info['user_id']).first()
        return {'msg':'ok', 'data':dict(new_user),'code':0}
    else:
        return {'msg':'fail,no such user_id'}


if __name__ == '__main__':
    app.run()
