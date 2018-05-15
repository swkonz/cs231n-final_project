
def print_lines(lines):
    f = open('cleansed_data.txt', 'a')
    for line in lines:
        if line == '': continue
        line = line.split('&')      
        if line[0] == '': continue
        if '' in line:
            stop_index = line.index('')
            line = line[:stop_index]
        f.write(' '.join(line))
        f.write('\n')
    f.close()

def read_data():
    lines = []
    with open("data1.txt") as f:
        for line in f:
            if line.startswith("&") or line == '' or line.startswith(' '): continue
            lines.append(line.strip())
    return lines

def main():
    lines = read_data()
    print_lines(lines)        

if __name__ == "__main__":
    main()