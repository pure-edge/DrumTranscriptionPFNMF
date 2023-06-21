import scipy.io

mat_file = 'DefaultSetting.mat'  # Replace with your .mat file path

data = scipy.io.loadmat(mat_file)
print(data['param'][0][0][0])