import os
import urllib

def main():
    downloader = urllib.URLopener()
    with open('cleansed_data.txt') as f:
        for num, line in enumerate(f):
            if num < 2454: continue
            line = line.split()
            dir_name = line[0].replace('/', '-')
            print 'Downloading ' + dir_name + ' (' + str(num) + ') videos.'
            try: 
                os.makedirs('./data/' + dir_name)
            except OSError:
                if not os.path.isdir('./data/'):
                    raise
            for index, url in enumerate(line[1:]):
                try:
                    downloader.retrieve(url, './data/' + dir_name + '/' + dir_name + str(index) + '.mov')
                except IOError:
                    continue
                except urllib.ContentTooShortError:
                    continue

if __name__ == "__main__":
    main()