import pyshark
from collections import defaultdict
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


pcap_path = r"C:\Users\chentian an\Desktop\leetcode\001.pcapng"
cap = pyshark.FileCapture(pcap_path,keep_packets=True)
cap.load_packets()
TCP_packet = []
UDP_packet = []
ICMP_packet = []
other_packet = []
protocols = defaultdict(list)
count = 0
udp_c = 0
ICMP_c = 0
o_c = 0
for packet in cap:
    try:
          # Determine and categorize protocols, save into an array
        if 'TCP' in packet:
            length = int(packet.length)
            src_ip = packet.ip.src if hasattr(packet, 'ip') else 'N/A'
            dst_ip = packet.ip.dst if hasattr(packet, 'ip') else 'N/A'
            time = packet.sniff_time
            TCP_packet.append((['TCP', length, src_ip, dst_ip,time]))

            count = count+1
        elif 'UDP' in packet:
            length = int(packet.length)
            src_ip = packet.ip.src if hasattr(packet, 'ip') else 'N/A'
            dst_ip = packet.ip.dst if hasattr(packet, 'ip') else 'N/A'
            time = packet.sniff_time
            UDP_packet.append((['UDP', length, src_ip, dst_ip,time]))
            udp_c += 1
        elif 'ICMP' in packet:
            length = int(packet.length)
            src_ip = packet.ip.src if hasattr(packet, 'ip') else 'N/A'
            dst_ip = packet.ip.dst if hasattr(packet, 'ip') else 'N/A'
            ICMP_packet.append((['ICMP', length, src_ip, dst_ip]))
            ICMP_c+=1
        else:
            length = int(packet.length)
            src_ip = packet.ip.src if hasattr(packet, 'ip') else 'N/A'
            dst_ip = packet.ip.dst if hasattr(packet, 'ip') else 'N/A'
            other_packet.append((['other', length, src_ip, dst_ip]))
            o_c +=1

    except Exception as e:
        print(f"Error processing packet: {e}")
    cap.close()

print("total tcp packet capture:",count,
      "total udp packet capture:",udp_c,
      "total icmp packet capture:",ICMP_c,
      "total other packet capture:",o_c)


class_packet = TCP_packet+UDP_packet

def ip_to_int(ip):
    try:
        return sum([int(num) << (8 * (3 - i)) for i, num in enumerate(ip.split("."))])
    except:
        return 0

X = []
y = []

for proto, length, src, dst,time in class_packet:
    features = [
        length,
        ip_to_int(src),
        ip_to_int(dst),
    ]
    X.append(features)
    y.append(0 if proto == 'TCP' else 1)  # TCP=0, UDP=1

# to NumPy array
X = np.array(X)
y = np.array(y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Decision Tree Training
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Testing and Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['TCP', 'UDP']))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
labels = ['TCP', 'UDP']

#  Use the seaborn heat map to display
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('predict value')
plt.ylabel('true value')
plt.title('confusion matrix')
plt.tight_layout()
plt.show()



