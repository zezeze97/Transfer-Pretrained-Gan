from tensorboard.backend.event_processing import event_accumulator
 
#加载日志数据
ea=event_accumulator.EventAccumulator('output/finetune/Flowers_our_AdaFM/G_4_D_2/monitoring/events.out.tfevents.1666579666.AI-ThinkStation-P920') 
ea.Reload()
print(ea.scalars.Keys())
fid_lst = ea.scalars.Items('fid/score')
print(fid_lst)
