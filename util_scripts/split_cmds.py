f = open('../commands/command_BraTS.sh')
c1 = f.readlines()
f = open('../commands/command_mindboggle.sh')
c2 = f.readlines()
f = open('../commands/command_inputs.sh')
c3 = f.readlines()

cmds = c1 + c2 + c3

num_files = int(len(cmds) / 30)
for i in range(num_files+1):
    f = open('../commands/commands_'+str(i)+'.sh', 'w')
    if i == num_files:
        f.writelines(cmds[i*30:])
    else:
        f.writelines(cmds[i * 30:(i+1) * 30])
    f.close()

print('done')