import requests as re
import time as t
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os.path

cached_state = None

MCTS_SIMULATIONS = 50
MAX_SKIPPED_TURNS = 50
MODEL_SAVE_PATH = "gomoku_model.pth"
SAVE_INTERVAL = 2

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()

        self.fc1 = nn.Linear(15 * 15, 128)  # 第一层
        self.fc2 = nn.Linear(128, 64)  # 第二层
        self.fc3 = nn.Linear(64, 1)  # 价值网络输出：局面的胜率
        self.fc4 = nn.Linear(64, 15 * 15)  # 策略网络输出：每个位置的概率

    def forward(self, x):
        x = x.view(-1, 15 * 15)  # 将棋盘展平为一维向量
        x = F.relu(self.fc1(x))  # 第一层激活
        x = F.relu(self.fc2(x))  # 第二层激活
        value = torch.tanh(self.fc3(x))  # 价值网络输出
        policy = F.softmax(self.fc4(x), dim=1)  # 策略网络输出
        return value, policy


model = AlphaZeroNet()


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
    url = 'http://183.175.12.27:8001/join_game/'
    param = {
        'user': user,
        'password': myHexPass,
        'data_type': 'json'
    }
    getHtml = re.get(url, params=param)
    print(f"Open a new game{getHtml.text}")
    return getHtml


def check_game(game_id):
    url = f'http://183.175.12.27:8001/check_game/{game_id}'
    getState = re.get(url)
    game_state = getState.json()
    board = game_state.get('board', [])
    if not board:
        print("The board is empty at the start of the game.")
    return game_state


def play_game(user, myHexPass, game_id, coord):
    url = f'http://183.175.12.27:8001/play_game/{game_id}'
    param = {
        'user': user,
        'password': myHexPass,
        'data_type': 'json',
        'coord': coord
    }
    re.get(url, params=param)

def getIndexNum(coords):
    if isinstance(coords, int):
        coords = str(coords)
    if len(coords) == 2 and isinstance(coords, str):
        return (ord(coords[0]) - ord('a')) * 16 + ord(coords[1]) - ord('a')
    else:
        raise ValueError(f"Invalid coordinates format: {coords}")

def getMaxCoords(state, RWP, indexSrc):
    if isinstance(state, list) and len(state) == 225:
        board = ''
        for i in range(15):
            for j in range(15):
                idx = i * 15 + j
                if state[idx] == 1:
                    board += 'M'
                elif state[idx] == -1:
                    board += 'O'
                else:
                    board += '.'
            board += '\n'
    else:
        Order = state
        board = ''
        for i in range(0, 15):
            board += '...............' + '\n'

        step = 0
        BW = judge(Order)
        for i in range(0, len(Order), 2):
            coords = Order[i:i + 2]
            index = getIndexNum(coords)
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
    return maxCoord if maxCoord else None

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

MCTS_SIMULATIONS = 50

def get_state(board):
    if isinstance(board, str):
        state = [0] * 225
        if board:
            BW = judge(board)

            for i in range(0, len(board), 2):
                if i + 1 < len(board):
                    index = getIndexNum(board[i:i + 2])
                    row = index // 16
                    col = index % 16
                    if 0 <= row < 15 and 0 <= col < 15:
                        pos = row * 15 + col
                        if pos < 225:
                            state[pos] = 1 if (i // 2) % 2 == 0 else -1
    else:
        state = []
        if not board:
            state = [0] * 225
        else:
            flat_board = []
            for row in board:
                if isinstance(row, list):
                    flat_board.extend(row)
                else:
                    flat_board.append(row)
            state = [1 if cell == 'X' or cell == 'M' else -1 if cell == 'O' else 0 for cell in flat_board]
            if len(state) < 225:
                state.extend([0] * (225 - len(state)))
            elif len(state) > 225:
                state = state[:225]
    print(f"State length: {len(state)}")
    return state

def get_possible_actions(state):
    if len(state) != 225:
        print(f"Warning: State length is {len(state)}, expected 225")
        return []
    return [i for i, cell in enumerate(state) if cell == 0]

def mcts_search(state):
    if len(state) != 225:
        print(f"Warning in mcts_search: State length is {len(state)}, expected 225")
        state = [0] * 225
    possible_actions = get_possible_actions(state)
    if not possible_actions:
        print("No possible actions found")
        return None, 0
    best_action = None
    best_value = -float('inf')
    for _ in range(MCTS_SIMULATIONS):
        action, value = evaluate_state(state)
        if action is not None and value > best_value:
            best_value = value
            best_action = action
    return best_action, best_value

def evaluate_state(state):
    if len(state) != 225:
        raise ValueError(f"Expected state of length 225, got {len(state)}")
    possible_actions = get_possible_actions(state)
    if not possible_actions:
        return None, 0
    board_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        value, policy = model(board_tensor)
    action_probs = policy[0].detach().numpy()
    valid_action_probs = []
    for action in possible_actions:
        if action < len(action_probs):
            valid_action_probs.append(action_probs[action])
        else:
            valid_action_probs.append(0.0)
    if not valid_action_probs:
        return None, 0
    best_idx = np.argmax(valid_action_probs)
    best_action = possible_actions[best_idx]
    return best_action, value.item()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_step(state, action, reward):
    model.train()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_tensor = torch.tensor([action], dtype=torch.long)
    value, policy = model(state_tensor)
    value_loss = criterion(value, torch.tensor([reward], dtype=torch.float32))
    policy_loss = -torch.log(policy[0, action_tensor]) * reward
    loss = value_loss + policy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def choose_action(state, RWP, indexSrc):
    action_coord = getMaxCoords(state, RWP, indexSrc)
    if action_coord:
        try:
            row = ord(action_coord[0]) - ord('a')
            col = ord(action_coord[1]) - ord('a')
            action = row * 15 + col
            print(f"Chosen action from heuristic strategy: {action_coord} (index: {action})")
        except (IndexError, TypeError):
            print(f"Invalid action coordinate: {action_coord}")
            action = None
    else:
        print("Heuristic strategy did not find a valid action. Falling back to neural network.")

        action, _ = mcts_search(state)
    if action is None:
        print("Exploring with random action due to no valid action found.")
        possible_actions = get_possible_actions(state)
        if possible_actions:
            action = random.choice(possible_actions)
        else:
            print("No possible actions available.")
            return None
    return action

def get_reward(state, action, winner):
    if winner == 'None':
        return 0
    if winner == user:
        return 1
    else:
        return -1

def save_model(model, optimizer, path=MODEL_SAVE_PATH):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer, path=MODEL_SAVE_PATH):
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print(f"No saved model found at {path}")
        return False

def train_agent(user, game_id):
    game_state = check_game(game_id)
    board = game_state.get('board', '')

    # 检查是否是空棋盘且轮到自己(黑棋)下棋
    if game_state.get('current_turn') == user and not board:
        # 黑棋先手，默认下在中央位置hh
        coord = 'hh'
        print(f"我方为黑棋，先手落子在棋盘中央({coord})")
        play_game(user, myHexPass, game_id, coord)
        print(f"✅ 已在位置 {coord} 落子")
        t.sleep(1.5)  # 等待服务器响应
        game_state = check_game(game_id)
        board = game_state.get('board', '')

    state_str = get_state(board)
    skipped_turns = 0
    training_steps = 0

    while True:
        # 检查是否轮到我们行动
        if game_state.get('current_turn', '') != user:
            print(f"等待对手落子...")
            waiting = True
            while waiting:
                t.sleep(2)
                game_state = check_game(game_id)
                if game_state.get('current_turn', '') == user or game_state.get('winner', 'None') != 'None':
                    waiting = False
                    board = game_state.get('board', '')
                    state_str = get_state(board)
            continue

        action = choose_action(state_str, RWP, indexSrc)
        if action is None:
            print("跳过回合：没有有效动作")
            skipped_turns += 1
            if skipped_turns > MAX_SKIPPED_TURNS:
                print("跳过回合次数过多，结束训练...")
                break
            t.sleep(2)
            game_state = check_game(game_id)
            board = game_state.get('board', '')
            state_str = get_state(board)
            continue

        row = action // 15
        col = action % 15
        coord = chr(97 + row) + chr(97 + col)
        print(f"落子位置：{coord} (action={action})")
        play_game(user, myHexPass, game_id, coord)
        t.sleep(1)

        new_game_state = check_game(game_id)
        new_board = new_game_state.get('board', '')
        new_state_str = get_state(new_board)
        winner = new_game_state.get('winner', 'None')
        reward = get_reward(state_str, action, winner)
        loss = train_step(state_str, action, reward)

        training_steps += 1
        print(f"训练步骤 {training_steps} - 损失: {loss}")
        if training_steps % SAVE_INTERVAL == 0:
            save_model(model, optimizer)

        if winner != 'None':
            print(f"赢家是 {winner}")
            save_model(model, optimizer)
            break

        state_str = new_state_str
        game_state = new_game_state
        skipped_turns = 0

user = 'name'
password = '12345'
myHexPass = encodeLogin(password)
RWP = RuleWithPoints()
indexSrc = allIndexStr()

load_model(model, optimizer)

game_id = join_game(user, myHexPass).json()["game_id"]
state = check_game(game_id)

print("Looking for game partners ...")
while state['ready'] == "False":
    state = check_game(game_id)
    print(state['ready'], end=" ")
    t.sleep(2)

if state['creator'] != user:
    opponent = state['creator']
else:
    opponent = state['opponent_name']

train_agent(user, game_id)

while state['ready'] == "True":
    if state['current_turn'] == user:
        order = state['board']
        coord = getMaxCoords(order, RWP, indexSrc)
        play_game(user, myHexPass, game_id, coord)
        print(f"Playing {coord}")
    else:
        print(f"Waiting for {opponent} to play")

    t.sleep(2)
    state = check_game(game_id)

    if state['winner'] != "None":
        print(f"The winner is {state['winner']}")
        break
