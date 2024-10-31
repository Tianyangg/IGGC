import visdom

# def draw_visdom(viz, train_x, train_loss, win_name):
#     viz.line(X=train_x, Y=train_loss, win=win_name, 
#              opts={
#                 'showlegend': True,  # 显示网格
#                 'title': win_name,
#                 'xlabel': "epoch",  # x轴标签
#                 'ylabel': "Loss",  },
#                 )
#     # viz.line(X=val_x, Y=val_loss, name="val_loss", update="append", win=win_name)

def draw_visdom(viz, train_x, train_loss, val_x, val_loss, win_name):
    viz.line(X=train_x, Y=train_loss, win=win_name, name="train_acc", 
             opts={
                'showlegend': True,  # 显示网格
                'title': win_name,
                'xlabel': "epoch",  # x轴标签
                'ylabel': "Acc",  },
                )
    viz.line(X=val_x, Y=val_loss, name="val_acc", update="append", win=win_name)