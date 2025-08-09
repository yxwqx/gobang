import requests as re
import time as t

cached_state = None

def fastModular(x):
    result = 1
    while (x[1] > 0):
        if (x[1] & 1):
            result = result * x[0] % x[2]
        x[1] = int(x[1] / 2)
        x[0] = x[0] * x[0] % x[2]
    return result

def str_to_num(strings):
    sum = 0
    lens = len(strings)
    for i in range(0, lens):
        sum += ord(strings[i]) * 256 ** (lens - i - 1)
    return sum

def encodeLogin(password):
    power = 65537
    modulus = 135261828916791946705313569652794581721330948863485438876915508683244111694485850733278569559191167660149469895899348939039437830613284874764820878002628686548956779897196112828969255650312573935871059275664474562666268163936821302832645284397530568872432109324825205567091066297960733513602409443790146687029
    return hex(fastModular([str_to_num(password), power, modulus]))

def join_game(user, myHexPass):
    url = 'http://183.175.12.27:8004/join_game/'
    param = {
        'user': user,
        'password': myHexPass,
        'data_type': 'json'
    }
    getHtml = re.get(url, params=param)
    print(f"Open a new game{getHtml.text}")
    return getHtml

def check_game(game_id):
    url = f'http://183.175.12.27:8004/check_game/{game_id}'
    getState = re.get(url)
    return getState

def play_game(user, myHexPass, game_id, coord):
    url = f'http://183.175.12.27:8004/play_game/{game_id}'
    param = {
        'user': user,
        'password': myHexPass,
        'data_type': 'json',
        'coord': coord
    }
    re.get(url, params=param)

def getIndexNum(coords):
    return (ord(coords[0]) - ord('a')) * 16 + ord(coords[1]) - ord('a')

def allIndexStr():
    spot = []
    for i in range(0, 15):
        for j in range(0, 16):
            spot.append(chr(i + 97) + chr(j + 97))
    return spot

def getLine(coord, board):
    line = ['', '', '', '']
    i = 0
    while (i != 15):
        if ord(coord[1]) - ord('a') - 7 + i in range(0, 15):
            line[0] += board[(ord(coord[0]) - ord('a')) * 16 + ord(coord[1]) - ord('a') - 7 + i]
        else:
            line[0] += ' '
        if ord(coord[0]) - ord('a') - 7 + i in range(0, 15):
            line[2] += board[(ord(coord[0]) - ord('a') - 7 + i) * 16 + ord(coord[1]) - ord('a')]
        else:
            line[2] += ' '
        if ord(coord[1]) - ord('a') - 7 + i in range(0, 15) and ord(coord[0]) - ord('a') - 7 + i in range(0, 15):
            line[1] += board[(ord(coord[0]) - ord('a') - 7 + i) * 16 + ord(coord[1]) - ord('a') - 7 + i]
        else:
            line[1] += ' '
        if ord(coord[1]) - ord('a') + 7 - i in range(0, 15) and ord(coord[0]) - ord('a') - 7 + i in range(0, 15):
            line[3] += board[(ord(coord[0]) - ord('a') - 7 + i) * 16 + ord(coord[1]) - ord('a') + 7 - i]
        else:
            line[3] += ' '
        i += 1
    return line

def judge(testOrder):
    if (len(testOrder) // 2) % 2 == 0:
        return 'MO'
    else:
        return 'OM'

def RuleWithPoints():
    RWP = {
        ("CMMMM", "MCMMM", "MMCMM", "MMMCM", "MMMMC"): 10000,
        ("COOOO", "OCOOO", "OOCOO", "OOOCO", "OOOOC"): 6000,
        (".CMMM.", ".MCMM.", ".MMCM.", ".MMMC."): 5000,
        ("COOO.", ".OOOC", ".OOCO.", ".OCOO."): 2500,
        ("OCMMM.", "OMCMM.", "OMMCM.", "OMMMC.", ".CMMMO", ".MCMMO", ".MMCMO", ".MMMCO"): 2000,
        (".MMC.", ".MCM.", ".CMM."): 400,
        (".OOC", "COO.", "MOOOC", "COOOM"): 400,
        (".MMCO", ".MCMO", ".CMMO", "OMMC.", "OMCM.", "OCMM.", "MOOC", "COOM"): 200,
        (".MC.", ".CM."): 50,
        ('.'): 1
    }
    return RWP

def getMaxCoords(Order, RWP, indexSrc):
    board = ''
    for i in range(0, 15):
        board += '...............' + '\n'
    step = 0
    BW = judge(Order)
    for i in range(0, len(Order), 2):
        index = getIndexNum(Order[i:i + 2])
        if (step % 2) == 0:
            board = board[0: index] + BW[0] + board[index + 1:]
        else:
            board = board[0: index] + BW[1] + board[index + 1:]
        step += 1
    maxCoord = ''
    maxPoints = 0
    for i in range(0, len(board)):
        if board[i] == '.':
            tempBoard = board[0: i] + 'C' + board[i + 1:]
            coord = indexSrc[i]
            lines4 = ','.join(getLine(coord, tempBoard))
            points = 0
            for rules, value in RWP.items():
                for rul in range(0, len(rules)):
                    if rules[rul] in lines4:
                        points += value * lines4.count(rules[rul])
            if points > maxPoints:
                maxPoints = points
                maxCoord = coord
    return maxCoord

user = 'name'
password = '12345'
myHexPass = encodeLogin(password)
RWP = RuleWithPoints()
indexSrc = allIndexStr()

game_id = join_game(user, myHexPass).json()["game_id"]
state = check_game(game_id).json()
print("Looking for game partners ...")
while state['ready'] == "False":
    state = check_game(game_id).json()
    print(state['ready'], end=" ")
    t.sleep(2)
if state['creator'] != user:
    opponent = state['creator']
else:
    opponent = state['opponent_name']

while state['ready'] == "True":
    if state['current_turn'] == user:
        order = state['board']
        coord = getMaxCoords(order, RWP, indexSrc)
        play_game(user, myHexPass, game_id, coord)
        print(f"Playing {coord}")
    else:
        print(f"Waiting for {opponent} to play")

    t.sleep(2)
    state = check_game(game_id).json()

    if state['winner'] != "None":
        print(f"The winner is {state['winner']}")
        break
