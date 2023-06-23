import os

def recursiveFileList(directoryList, extName, maxFileNumInEachDir=float('inf')):
    # RecursiveFileList: List files with a given extension recursively
    # Usage: allData=recursiveFileList(directoryList, extName, maxFileNumInEachDir)

    if not directoryList:
        raise ValueError('Need at least one input argument!')
    if not isinstance(directoryList, list):
        directoryList = [directoryList]

    allData = []
    for directory in directoryList:
        if directory.endswith('/') or directory.endswith('\\'):
            directory = directory[:-1]

        # Get files in the given directory
        data = os.scandir(directory)
        data = [d for d in data if d.is_file() and d.name.endswith('.' + extName)]
        data = data[:min(len(data), maxFileNumInEachDir)]

        '''for i in range(len(data)):
            data[i].path = os.path.join(directory, data[i].name)
            parentPath, _, _ = os.path.split(data[i].path)
            _, data[i].parentDir, _ = os.path.split(parentPath)'''

        # Get files in sub-directories
        subdirs = os.scandir(directory)
        subdirs = [d for d in subdirs if d.is_dir()]
        for subdir in subdirs:
            if subdir.name in ('.', '..'):
                continue
            thisPath = os.path.join(directory, subdir.name)
            data2 = recursiveFileList(thisPath, extName, maxFileNumInEachDir)
            if not data2:
                data2 = []
            if not data:
                data = []
            data += data2

        allData += data

    return allData
