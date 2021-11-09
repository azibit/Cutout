import csv, torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

def get_pred_as_list(test_preds):

    result = []
    for i in range(len(test_preds)):
        result.append(int(test_preds[i].item()))

    return result

def make_prediction(net, class_names, loader, name_to_save):

    test_loss = 0
    correct = 0
    total = 0
    acc = 0
    all_preds = torch.tensor([]).cuda()
    ground_truths = torch.tensor([]).cuda()
    net.eval()

    for batch_idx, (inputs, targets) in enumerate(loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            ground_truths = torch.cat(
                  (ground_truths, targets)
                  ,dim=0
              )

            all_preds = torch.cat(
                  (all_preds, predicted)
                  ,dim=0
              )

            acc = 100.*correct/total

    targets = get_pred_as_list(ground_truths)
    preds = get_pred_as_list(all_preds)
    cm = confusion_matrix(targets, preds)
    return classification_report(targets, preds, target_names=class_names)
