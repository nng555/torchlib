import numpy as np
import math
import argparse

# softmax outputs to probabilities
def softmax(probs):
   esum = sum([math.exp(x) for x in probs])
   return [math.exp(x)/esum for x in probs]

# generate DP likelihood errors for each subsequence
def likelihood_err(prob, n):
   err = np.zeros((n,n))
   for i in range(n):
      err[i][i] = -np.log(prob[i])
   for i in range(n):
      for j in range(i+1, n):
         if i == 0:
            err[i][j] = err[i][j-1] - np.log(prob[j])
         else:
            err[i][j] = err[i-1][j] - err[i-1][j-1] + err[i][j-1]
   return err

# generate DP earthmover errors for each subsequence
def earthmover_err(prob, n, stage):
   err = np.zeros((n,n))
   stagenums = np.asarray(range(6))
   for i in range(n):
      err[i][i] = np.sum(prob[i] * np.abs(stagenums-stage))
   for i in range(n):
      for j in range(i+1, n):
         if i == 0:
            err[i][j] = err[i][j-1] + np.sum(prob[j] * np.abs(stagenums - stage))
         else:
            err[i][j] = err[i-1][j] - err[i-1][j-1] + err[i][j-1]
   return err

# use DP to find split points that minimize total error
def find_splits(stages, n, metric):
   err = []
   for i in range(n):
      if metric == "likelihood":
         err.append(likelihood_err(stages[:,i], len(stages)))
      elif metric == "earthmover":
         err.append(earthmover_err(stages, len(stages), i))
   stageErr = np.zeros((n, len(stages)))
   prevSplit = np.zeros((n, len(stages)))
   for i in range(n):
      for j in range(1, len(stages)):
         if i == 0:
            stageErr[i][j] = err[i][0][j]
            prevSplit[i][j] = j
         else:
            tempErr = []
            for k in range(j):
               tempErr.append(stageErr[i-1][k] + err[i][k][j])
            prevSplit[i][j] = tempErr.index(min(tempErr))
            stageErr[i][j] = min(tempErr)
   splits = []
   for i in range(n):
      if len(splits) == 0:
         splits.append(prevSplit[n-i-1][len(stages)-1])
      else:
         splits.append(prevSplit[n-i-1][int(splits[-1])])
   return splits[::-1][1:]

# turn splits into array of predictions
def split_pred(splits, n, size):
   res = []
   for i in range(size):
      for j in range(n):
         if i > splits[-1]:
            res.append(n-1)
            break
         if i <= splits[j]:
            res.append(j)
            break
   return res

def eval_metric(target, stages, pred_splits, metric):
   # evaluate likelihood splits
   splits = []
   preds = []
   last = 0

   for i in range(len(target)):
      if i == len(target) - 1 or target[i] > target[i+1]:
         split = find_splits(stages[last:i], 6, metric)
         pred = split_pred(split, 6, i-last)
         splits.append(split)
         preds.extend(pred)
         last = i
   preds.append(5)

   preds = np.asarray(preds)
   splits = np.asarray(splits)

   pf_acc = np.sum(np.equal(target, preds)) * 1.0/len(target)
   ps_mae = np.sum(np.abs(splits - pred_splits), axis=0) / (1.0 * len(pred_splits))
   ps_rmse = np.sqrt(np.sum((splits - pred_splits)**2, axis=0) / (1.0 * len(pred_splits)))

   return pf_acc, ps_mae, ps_rmse, preds

def smooth(conf):

   # load raw data
   model_dir = (
         '/data2/nathan/embryo/model/' + conf['conf_file'] + '/' +
         str(conf['lr']) + 'lr_' + str(conf['in_ch']) + 'inch_' + str(conf['optim']) + '/'
   )
   raw = np.load(model_dir + 'probs' + str(conf['nb_epoch']) + conf['eval'] + '.npy')
   if 'trans' in conf['conf_file']:
      raw = raw[:,:-1]
   target = np.load('/data2/nathan/' + conf['dataset'] + '/' + conf['eval'] + '/labelCutIndex.npy')
   stages = np.asarray([softmax(x) for x in raw])

   # evaluate raw outputs
   raw_pred = np.asarray([np.argmax(x) for x in stages])
   raw_pf_acc = np.sum(np.equal(target, raw_pred)) * 1.0/len(target)

   # find target splits
   last = 0
   pred_splits = []
   tmp = []
   for i in range(len(target)):
      if i == len(target) - 1 or target[i] > target[i+1]:
         while(len(tmp) != 5):
            tmp.append(-1)
         pred_splits.append(tmp)
         tmp = []
         last = i
      if i != len(target) - 1:
         for j in range(target[i+1] - target[i]):
            tmp.append(i-last)
   pred_splits = np.asarray(pred_splits)

   ll_pf_acc, ll_ps_mae, ll_ps_rmse, ll_preds = eval_metric(target, stages, pred_splits, "likelihood")
   earth_pf_acc, earth_ps_mae, earth_ps_rmse, earth_preds = eval_metric(target, stages, pred_splits, "earthmover")

   print("Raw: Per Frame Accuracy: %.4f" %(raw_pf_acc))
   print("Likelihood: Per Frame Accuracy: %.4f" %(ll_pf_acc))
   print("Earthmover: Per Frame Accuracy: %.4f" %(earth_pf_acc))
   print("Likelihood: Per Stage MAE: " + str(ll_ps_mae) + " Overall MAE: " + str(np.mean(ll_ps_mae)))
   print("Earthmover: Per Stage MAE: " + str(earth_ps_mae) + " Overall MAE: " + str(np.mean(earth_ps_mae)))
   print("Likelihood: Per Stage RMSE: " + str(ll_ps_rmse) + " Overall RMSE: " + str(np.sqrt(np.mean(ll_ps_rmse**2))))
   print("Earthmover: Per Stage RMSE: " + str(earth_ps_rmse) + " Overall RMSE: " + str(np.sqrt(np.mean(earth_ps_rmse**2))))

   np.save(model_dir + 'raw_pred' + str(conf['nb_epoch']) + '.npy', raw_pred)
   np.save(model_dir + 'll_pred' + str(conf['nb_epoch']) + '.npy', ll_preds)
   np.save(model_dir + 'earth_pred' + str(conf['nb_epoch']) + '.npy', earth_preds)

if __name__=="__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-d', action='store', dest='dataset', type=str,
         help='the dataset to train on')
   parser.add_argument('-lr', action='store', dest='lr', type=float,
         help='the learning rate')
   parser.add_argument('-n', action='store', dest='nb_epoch', type=int, default=300,
         help='the epoch to evaluate at')
   parser.add_argument('-f', action='store', dest='conf_file', type=str,
         help='the architecture config file')
   parser.add_argument('-i', action='store', dest='in_ch', type=int,
         help='the number of channel inputs', default=3)
   parser.add_argument('-o', action='store', dest='optim', type=str,
         help='the optimizer to use', default='SGD')
   parser.add_argument('-e', action='store', dest='eval', type=str, default='val',
         help='the data split to evaluate on')
   args = vars(parser.parse_args())

   smooth(args)
