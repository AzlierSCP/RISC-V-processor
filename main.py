import os
import argparse
import shutil

MemSize = 1000  # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.


class InsMem(object):
    def __init__(self, name, ioDir):
        self.id = name

        with open(ioDir + os.sep + "imem.txt", "r") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def readInstr(self, ReadAddress):
        # read instruction memory
        # return 32 bit hexadecimal value
        inst = int("".join(self.IMem[ReadAddress: ReadAddress + 4]), 2)
        return format(inst, '#010x')

    def readInstr_FS(self, ReadAddress_FS: int) -> str:
        # read instruction memory
        # return 32 bit binary value
        return "".join(self.IMem[ReadAddress_FS: ReadAddress_FS + 4])


class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + os.sep + "dmem.txt", "r") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]
        self.DMem = self.DMem + (['00000000'] * (MemSize - len(self.DMem)))

    def readDM(self, ReadAddress):
        # read data memory
        # return 32 bit hex
        data32 = int("".join(self.DMem[ReadAddress: ReadAddress + 4]), 2)  # change into decimal number
        return format(data32, '#010x')  # '0x'+8 bit hex

    def readDM_FS(self, read_addr: str) -> str:
        # read data memory
        # return 32 bit hex val
        read_addr_int = bin2int(read_addr)
        return "".join(self.DMem[read_addr_int: read_addr_int + 4])

    def writeDM(self, Address, WriteData):
        # write data into byte addressable memory
        mask8 = int('0b11111111', 2)  # 8-bit mask
        data8_arr = []

        for j in range(4):
            data8_arr.append(WriteData & mask8)
            WriteData = WriteData >> 8

        for i in range(4):
            # most significant bit(last element in data8_arr) in smallest address
            self.DMem[Address + i] = format(data8_arr.pop(), '08b')

    def writeDM_FS(self, addr: str, write_data: str):
        # write data into byte addressable memory
        addr_int = bin2int(addr)
        for i in range(4):
            self.DMem[addr_int + i] = write_data[8 * i: 8 * (i + 1)]

    # output file of Dmem  SS_DMEMResult.txt
    def outputDataMem(self):
        resPath = self.ioDir + os.sep + self.id + "_DMEMResult.txt"
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])


class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = [0x0 for i in range(32)]
        self.RegistersFS = [int2bin(0) for _ in range(32)]

    def readRF(self, Reg_addr):
        return self.Registers[Reg_addr]

    def writeRF(self, Reg_addr, Wrt_reg_data):
        if Reg_addr != 0:
            self.Registers[Reg_addr] = Wrt_reg_data & ((1 << 32) - 1)  # and 32 bits 1 mask

    def outputRF(self, cycle):
        op = ["State of RF after executing cycle:  " + str(cycle) + "\n"]
        op.extend([format(val, '032b') + "\n" for val in self.Registers])
        if cycle == 0:
            perm = "w"
        else:
            perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

    # five stage
    def readRF_FS(self, reg_addr: str) -> str:
        return self.RegistersFS[bin2int(reg_addr)]

    def writeRF_FS(self, reg_addr: str, wrt_reg_data: str):
        if reg_addr == "00000":
            return
        self.RegistersFS[bin2int(reg_addr)] = wrt_reg_data

    def outputRF_FS(self, cycle):
        op = ["State of RF after executing cycle:" + str(cycle) + "\n"]
        op.extend([f"{val}" + "\n" for val in self.RegistersFS])
        if cycle == 0:
            perm = "w"
        else:
            perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)


def int2bin(x: int, n_bits: int = 32) -> str:
    bin_x = bin(x & (2 ** n_bits - 1))[2:]
    return "0" * (n_bits - len(bin_x)) + bin_x


def bin2int(x: str, sign_ext: bool = False) -> int:
    x = str(x)
    if sign_ext and x[0] == "1":
        return -(-int(x, 2) & (2 ** len(x) - 1))
    return int(x, 2)


class Core(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.inst = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem


class State(object):
    def __init__(self):
        # Instruction Fetch (IF) Stage Registers
        self.IF = {
            "nop": bool(False),
            "PC": int(0),
            "taken": bool(False)
        }

        # Instruction Decode (ID) Stage Registers
        self.ID = {
            "nop": bool(False),
            "instr": str("0"*32),
            "PC": int(0),
            "hazard_nop": bool(False)
        }

        # Execution (EX) Stage Registers
        self.EX = {
            "nop": bool(False),
            "instr": str("0"*32),
            "Read_data1": str("0"*32),
            "Read_data2": str("0"*32),
            "Imm": str("0"*32),
            "Rs": str("0"*5),
            "Rt": str("0"*5),
            "Wrt_reg_addr": str("0"*5),
            "is_I_type": bool(False),
            "rd_mem": bool(False),
            "wrt_mem": bool(False),
            "alu_op": str("00"),
            "wrt_enable": bool(False)
        }

        # Memory Access (MEM) Stage Registers
        self.MEM = {
            "nop": bool(False),
            "ALUresult": str("0"*32),
            "Store_data": str("0"*32),
            "Rs": str("0"*5),
            "Rt": str("0"*5),
            "Wrt_reg_addr": str("0"*5),
            "rd_mem": bool(False),
            "wrt_mem": bool(False),
            "wrt_enable": bool(False)
        }

        # Write Back (WB) Stage Registers
        self.WB = {
            "nop": bool(False),
            "Wrt_data": str("0"*32),
            "Rs": str("0"*5),
            "Rt": str("0"*5),
            "Wrt_reg_addr": str("0"*5),
            "wrt_enable": bool(False)
        }


class Core_FS(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.inst_FS = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State_FS()
        self.nextState = State_FS()
        self.ext_imem = imem
        self.ext_dmem = dmem


class State_FS(object):
    def __init__(self):
        self.IF = IF_State()
        self.ID = ID_State()
        self.EX = EX_State()
        self.MEM = MEM_State()
        self.WB = WB_State()

    def next(self):
        self.ID = ID_State()
        self.EX = EX_State()
        self.MEM = MEM_State()
        self.WB = WB_State()


class HazardDetector:
    def __init__(self, state: State_FS):
        self.state = state

    def detect(self, rs):
        # EX to 1
        if rs == self.state.MEM.write_reg_addr and self.state.MEM.read_mem == 0:
            return 2
        # EX to 2 / MEM to 2
        elif rs == self.state.WB.write_reg_addr and self.state.WB.write_enable:
            return 1
        # MEM to 1
        elif rs == self.state.MEM.write_reg_addr and self.state.MEM.read_mem != 0:
            self.state.ID.hazard_nop = True
            return 1
        else:
            return 0


def sign_extend(val, sign_bit):

    if (val & (1 << sign_bit)) != 0:
        val = val - (1 << (sign_bit + 1))
    return val


def ALU_I(funct3, rs1, imm):
    rd = 0
    # ADDI
    if funct3 == 0b000:
        rd = rs1 + sign_extend(imm, 11)

    # XORI
    if funct3 == 0b100:
        rd = rs1 ^ sign_extend(imm, 11)

    # ORI
    if funct3 == 0b110:
        rd = rs1 | sign_extend(imm, 11)

    # ANDI
    if funct3 == 0b111:
        rd = rs1 & sign_extend(imm, 11)

    return rd


def ALU_R(funct7, funct3, rs1, rs2):
    rd = 0
    # ADD
    if funct7 == 0b0000000 and funct3 == 0b000:
        rd = rs1 + rs2

    # SUB
    if funct7 == 0b0100000 and funct3 == 0b000:
        rd = rs1 - rs2

    # XOR
    if funct7 == 0b0000000 and funct3 == 0b100:
        rd = rs1 ^ rs2

    # OR
    if funct7 == 0b0000000 and funct3 == 0b110:
        rd = rs1 | rs2

    # AND
    if funct7 == 0b0000000 and funct3 == 0b111:
        rd = rs1 & rs2

    return rd


class IF_State:
    def __init__(self) -> None:
        self.nop: bool = False
        self.PC: int = 0

    def __dict__(self):
        return {
            "PC": self.PC,
            "nop": self.nop
        }


class ID_State:
    def __init__(self) -> None:
        self.nop: bool = True
        self.hazard_nop: bool = False
        self.PC: int = 0
        self.instr: str = "0" * 32

    def __dict__(self):
        return {
            "Instr": self.instr[::-1],
            "nop": self.nop
        }


class EX_State:
    def __init__(self) -> None:
        self.nop: bool = True
        self.instr: str = ""
        self.read_data_1: str = "0" * 32
        self.read_data_2: str = "0" * 32
        self.imm: str = "0" * 32
        self.rs: str = "0" * 5
        self.rt: str = "0" * 5
        self.write_reg_addr: str = "0" * 5
        self.is_I_type: bool = False
        self.read_mem: bool = False
        self.write_mem: bool = False
        self.alu_op: str = "00"
        self.write_enable: bool = False

    def __dict__(self):
        return {
            "nop": self.nop,
            "instr": self.instr[::-1],
            "Operand1": self.read_data_1,
            "Operand2": self.read_data_2,
            "Imm": self.imm,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "is_I_type": int(self.is_I_type),
            "rd_mem": int(self.read_mem),
            "wrt_mem": int(self.write_mem),
            "alu_op": "".join(list(map(str, self.alu_op))),
            "wrt_enable": int(self.write_enable)
        }


class MEM_State:
    def __init__(self) -> None:
        self.nop: bool = True
        self.alu_result: str = "0" * 32
        self.store_data: str = "0" * 32
        self.rs: str = "0" * 5
        self.rt: str = "0" * 5
        self.write_reg_addr: str = "0" * 5
        self.read_mem: bool = False
        self.write_mem: bool = False
        self.write_enable: bool = False

    def __dict__(self):
        return {
            "nop": self.nop,
            "ALUresult": self.alu_result,
            "Store_data": self.store_data,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "rd_mem": int(self.read_mem),
            "wrt_mem": int(self.write_mem),
            "wrt_enable": int(self.write_enable)
        }


class WB_State:
    def __init__(self) -> None:
        self.nop: bool = True
        self.write_data: str = "0" * 32
        self.rs: str = "0" * 5
        self.rt: str = "0" * 5
        self.write_reg_addr: str = "0" * 5
        self.write_enable: bool = False

    def __dict__(self):
        return {
            "nop": self.nop,
            "Wrt_data": self.write_data,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "wrt_enable": int(self.write_enable)
        }


class IF_Stage:
    def __init__(self, state: State_FS, ins_mem: InsMem):
        self.state = state
        self.ins_mem = ins_mem

    def run(self):
        if self.state.IF.nop or self.state.ID.nop or (self.state.ID.hazard_nop and self.state.EX.nop):
            return
        instr = self.ins_mem.readInstr_FS(self.state.IF.PC)[::-1]
        if instr == "1" * 32:
            self.state.IF.nop = True
            self.state.ID.nop = True
        else:
            self.state.ID.PC = self.state.IF.PC
            self.state.IF.PC += 4
            self.state.ID.instr = instr


class ID_Stage:
    def __init__(self, state: State_FS, rf: RegisterFile, hazard_detector: HazardDetector):
        self.state = state
        self.rf = rf
        self.hazard_detector = hazard_detector

    def read_data(self, rs, forward_signal):
        if forward_signal == 1:
            return self.state.WB.write_data
        elif forward_signal == 2:
            return self.state.MEM.alu_result
        else:
            return self.rf.readRF_FS(rs)

    def run(self):
        if self.state.ID.nop:
            if not self.state.IF.nop:
                self.state.ID.nop = False
            return

        self.state.EX.instr = self.state.ID.instr
        self.state.EX.is_I_type = False
        self.state.EX.read_mem = False
        self.state.EX.write_mem = False
        self.state.EX.write_enable = False
        self.state.ID.hazard_nop = False
        self.state.EX.write_reg_addr = "000000"
        opcode = self.state.ID.instr[:7][::-1]
        func3 = self.state.ID.instr[12:15][::-1]

        # R Type
        if opcode == "0110011":
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]
            forward_signal_1 = self.hazard_detector.detect(rs1)
            forward_signal_2 = self.hazard_detector.detect(rs2)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.read_data_2 = self.read_data(rs2, forward_signal_2)
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.write_enable = True
            func7 = self.state.ID.instr[25:][::-1]

            if func3 == "000":                 # add / sub
                self.state.EX.alu_op = "00"
                if func7 == "0100000":
                    self.state.EX.read_data_2 = int2bin(-bin2int(self.state.EX.read_data_2, sign_ext=True))
            elif func3 == "100":               # xor
                self.state.EX.alu_op = "11"
            elif func3 == "110":               # or
                self.state.EX.alu_op = "10"
            elif func3 == "111":               # and
                self.state.EX.alu_op = "01"

        # I Type
        elif opcode == "0010011" or opcode == "0000011":
            rs1 = self.state.ID.instr[15:20][::-1]
            forward_signal_1 = self.hazard_detector.detect(rs1)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.is_I_type = True
            self.state.EX.imm = self.state.ID.instr[20:][::-1]
            self.state.EX.write_enable = True
            self.state.EX.read_mem = opcode == "0000011"

            if func3 == "000":   # add
                self.state.EX.alu_op = "00"
            elif func3 == "100": # xor
                self.state.EX.alu_op = "11"
            elif func3 == "110": # or
                self.state.EX.alu_op = "10"
            elif func3 == "111": # and
                self.state.EX.alu_op = "01"

        # S W
        elif opcode == "0100011":
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]
            forward_signal_1 = self.hazard_detector.detect(rs1)
            forward_signal_2 = self.hazard_detector.detect(rs2)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.read_data_2 = self.read_data(rs2, forward_signal_2)
            self.state.EX.imm = (self.state.ID.instr[7:12] + self.state.ID.instr[25:])[::-1]
            self.state.EX.is_I_type = True
            self.state.EX.write_mem = True
            self.state.EX.alu_op = "00"

        # B Type
        elif opcode == "1100011":
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]
            forward_signal_1 = self.hazard_detector.detect(rs1)
            forward_signal_2 = self.hazard_detector.detect(rs2)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.read_data_2 = self.read_data(rs2, forward_signal_2)
            diff = bin2int(self.state.EX.read_data_1, sign_ext=True) - bin2int(self.state.EX.read_data_2, sign_ext=True)
            self.state.EX.imm = ("0" + self.state.ID.instr[8:12] + self.state.ID.instr[25:31]
                                 + self.state.ID.instr[7] + self.state.ID.instr[31])[::-1]

            if (diff == 0 and func3 == "000") or (diff != 0 and func3 == "001"):
                self.state.IF.PC = self.state.ID.PC + bin2int(self.state.EX.imm, sign_ext=True)
                self.state.ID.nop = True
                self.state.EX.nop = True
            else:
                self.state.EX.nop = True

        # J Type
        elif opcode == "1101111":
            self.state.EX.imm = ("0" + self.state.ID.instr[21:31] + self.state.ID.instr[20]
                                 + self.state.ID.instr[12:20] + self.state.ID.instr[31])[::-1]
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.read_data_1 = int2bin(self.state.ID.PC)
            self.state.EX.read_data_2 = int2bin(4)
            self.state.EX.write_enable = True
            self.state.EX.alu_op = "00"
            self.state.IF.PC = self.state.ID.PC + bin2int(self.state.EX.imm, sign_ext=True)
            self.state.ID.nop = True

        if self.state.IF.nop:
            self.state.ID.nop = True
        return 1


class EX_Stage:
    def __init__(self, state: State_FS):
        self.state = state

    def run(self):
        if self.state.EX.nop:
            if not self.state.ID.nop:
                self.state.EX.nop = False
            return

        operand_1 = self.state.EX.read_data_1
        operand_2 = (self.state.EX.read_data_2 if not self.state.EX.is_I_type and not self.state.EX.write_mem else self.state.EX.imm)

        if self.state.EX.alu_op == "00":
            self.state.MEM.alu_result = int2bin(bin2int(operand_1, sign_ext=True) + bin2int(operand_2, sign_ext=True))
        elif self.state.EX.alu_op == "01":
            self.state.MEM.alu_result = int2bin(bin2int(operand_1, sign_ext=True) & bin2int(operand_2, sign_ext=True))
        elif self.state.EX.alu_op == "10":
            self.state.MEM.alu_result = int2bin(bin2int(operand_1, sign_ext=True) | bin2int(operand_2, sign_ext=True))
        elif self.state.EX.alu_op == "11":
            self.state.MEM.alu_result = int2bin(bin2int(operand_1, sign_ext=True) ^ bin2int(operand_2, sign_ext=True))

        self.state.MEM.rs = self.state.EX.rs
        self.state.MEM.rt = self.state.EX.rt
        self.state.MEM.read_mem = self.state.EX.read_mem
        self.state.MEM.write_mem = self.state.EX.write_mem
        if self.state.EX.write_mem:
            self.state.MEM.store_data = self.state.EX.read_data_2
        self.state.MEM.write_enable = self.state.EX.write_enable
        self.state.MEM.write_reg_addr = self.state.EX.write_reg_addr
        if self.state.ID.nop:
            self.state.EX.nop = True


class MEM_Stage:
    def __init__(self, state: State_FS, data_mem: DataMem):
        self.state = state
        self.data_mem = data_mem

    def run(self):
        if self.state.MEM.nop:
            if not self.state.EX.nop:
                self.state.MEM.nop = False
            return

        if self.state.MEM.read_mem != 0:
            self.state.WB.write_data = self.data_mem.readDM_FS(self.state.MEM.alu_result)
        elif self.state.MEM.write_mem != 0:
            self.data_mem.writeDM_FS(self.state.MEM.alu_result, self.state.MEM.store_data)
        else:
            self.state.WB.write_data = self.state.MEM.alu_result
            self.state.MEM.store_data = self.state.MEM.alu_result
        self.state.WB.write_enable = self.state.MEM.write_enable
        self.state.WB.write_reg_addr = self.state.MEM.write_reg_addr

        if self.state.EX.nop:
            self.state.MEM.nop = True


class WB_Stage:
    def __init__(self, state: State_FS, rf: RegisterFile):
        self.state = state
        self.rf = rf

    def run(self):
        if self.state.WB.nop:
            if not self.state.MEM.nop:
                self.state.WB.nop = False
            return

        if self.state.WB.write_enable:
            self.rf.writeRF_FS(self.state.WB.write_reg_addr, self.state.WB.write_data)

        if self.state.MEM.nop:
            self.state.WB.nop = True


class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(ioDir + os.sep + "SS_", imem, dmem)
        self.opFilePath = ioDir + os.sep + "StateResult_SS.txt"

    def step(self):
        # implementation of each instruction
        fetchedInstr = int(self.ext_imem.readInstr(self.state.IF["PC"]), 16)  # hex into integer
        opcode = fetchedInstr & (2 ** 7 - 1)  # least significant 7 bits

        # decode and then execute
        self.Decode(opcode, fetchedInstr)

        self.halted = False
        if self.state.IF["nop"]:
            self.halted = True

        if not self.state.IF["taken"] and self.state.IF["PC"] + 4 < len(self.ext_imem.IMem):
            self.nextState.IF["PC"] = self.state.IF["PC"] + 4
        else:
            self.state.IF["taken"] = False

        self.myRF.outputRF(self.cycle)  # output file of registers (each cycle)
        self.printState(self.nextState, self.cycle)  # print states (each cycle)

        self.state = self.nextState
        self.cycle += 1
        self.inst += 1  # Count instructions

    def Decode(self, opcode, inst):
        # R-type
        if opcode == 0b0110011:

            # func7
            funct7 = inst >> 25
            # func3
            funct3 = (inst >> 12) & ((1 << 3) - 1)
            # rs2
            rs2 = (inst >> 20) & ((1 << 5) - 1)
            # rs1
            rs1 = (inst >> 15) & ((1 << 5) - 1)
            # rd
            rd = (inst >> 7) & ((1 << 5) - 1)

            # rs1
            data_rs1 = self.myRF.readRF(rs1)
            # rs2
            data_rs2 = self.myRF.readRF(rs2)
            # result data
            data_rd = ALU_R(funct7, funct3, data_rs1, data_rs2)
            # store all fetched and computed data
            self.myRF.writeRF(rd, data_rd)

        # I-type
        elif opcode == 0b0010011:
            # imm
            imm = inst >> 20 & ((1 << 12) - 1)

            # func3
            funct3 = (inst >> 12) & ((1 << 3) - 1)
            # rs1
            rs1 = (inst >> 15) & ((1 << 5) - 1)
            # rd
            rd = (inst >> 7) & ((1 << 5) - 1)

            # data in rs1
            data_rs1 = self.myRF.readRF(rs1)
            # result data
            data_rd = ALU_I(funct3, data_rs1, imm)
            # store result in rd
            self.myRF.writeRF(rd, data_rd)

        # S-type
        elif opcode == 0b0100011:

            # imm
            imm11_5 = inst >> 25
            imm4_0 = (inst >> 7) & ((1 << 5) - 1)
            imm = (imm11_5 << 5) | imm4_0

            # funct3
            funct3 = inst & (((1 << 3) - 1) << 12)
            # rs1
            rs1 = (inst >> 15) & ((1 << 5) - 1)
            # rd
            rs2 = (inst >> 20) & ((1 << 5) - 1)

            self.ext_dmem.writeDM(Address=(rs1 + sign_extend(imm, 11)) & ((1 << 32) - 1),
                                  WriteData=self.myRF.readRF(rs2))

        # B-type
        elif opcode == 0b1100011:

            # imm
            imm11 = (inst >> 7) & 1
            imm4_1 = (inst >> 8) & ((1 << 4) - 1)
            imm10_5 = (inst >> 25) & ((1 << 6) - 1)
            imm12 = (inst >> 31) & 1
            imm = (imm11 << 11) | (imm4_1 << 1) | (imm10_5 << 5) | (imm12 << 12)

            # rs2
            rs2 = (inst >> 20) & ((1 << 5) - 1)
            # rs1
            rs1 = (inst >> 15) & ((1 << 5) - 1)
            # funct3
            funct3 = (inst >> 12) & ((1 << 3) - 1)

            # BEQ
            if funct3 == 0b000:
                data_rs1 = self.myRF.readRF(rs1)
                data_rs2 = self.myRF.readRF(rs2)
                if data_rs1 == data_rs2:
                    self.nextState.IF["PC"] = self.state.IF["PC"] + sign_extend(imm, 12)
                    self.state.IF["taken"] = True

            # BNE
            else:
                data_rs1 = self.myRF.readRF(rs1)
                data_rs2 = self.myRF.readRF(rs2)
                if data_rs1 != data_rs2:
                    self.nextState.IF["PC"] = self.state.IF["PC"] + sign_extend(imm, 12)
                    self.state.IF["taken"] = True

        # J-type
        elif opcode == 0b1101111:
            # Jal
            # imm
            imm19_12 = (inst >> 12) & ((1 << 8) - 1)
            imm11 = (inst >> 20) & 1
            imm10_1 = (inst >> 21) & ((1 << 10) - 1)
            imm20 = (inst >> 31) & 1
            imm = (imm20 << 20) | (imm10_1 << 1) | (imm11 << 11) | (imm19_12 << 12)

            # rd
            rd = (inst >> 7) & ((1 << 5) - 1)

            self.myRF.writeRF(rd, self.state.IF["PC"] + 4)
            self.nextState.IF["PC"] = self.state.IF["PC"] + sign_extend(imm, 20)
            self.state.IF["taken"] = True

        # LW
        elif opcode == 0b0000011:

            # imm
            imm = inst >> 20
            # rs1
            rs1 = (inst >> 15) & ((1 << 5) - 1)
            # rd
            rd = (inst >> 7) & ((1 << 5) - 1)

            self.myRF.writeRF(Reg_addr=rd,
                              Wrt_reg_data=int(self.ext_dmem.readDM(
                                  ReadAddress=self.myRF.readRF(rs1) + sign_extend(imm, 11)), 16))

        # HALT
        else:
            self.state.IF["nop"] = True

    # Function to print the state after each execution cycle
    def printState(self, state, cycle):
        printstate = ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")

        if cycle == 0:
            perm = "w"
        else:
            perm = "a"

        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


class FiveStageCore(Core_FS):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + os.sep + "FS_", imem, dmem)
        self.opFilePath = ioDir + os.sep + "StateResult_FS.txt"
        self.hazard_detector = HazardDetector(self.state)
        self.if_stage = IF_Stage(self.state, self.ext_imem)
        self.id_stage = ID_Stage(self.state, self.myRF, self.hazard_detector)
        self.ex_stage = EX_Stage(self.state)
        self.mem_stage = MEM_Stage(self.state, self.ext_dmem)
        self.wb_stage = WB_Stage(self.state, self.myRF)

    def step(self):
        #  1  2  3  4  5  6  7  ...
        #  IF ID EX ME WB
        #     IF ID EX ME WB
        #        IF ID EX ME WB
        if (self.state.IF.nop
           and self.state.ID.nop
           and self.state.EX.nop
           and self.state.MEM.nop
           and self.state.WB.nop):
            self.halted = True

        current_instr = self.state.ID.instr
        # --------------------- WB stage ---------------------
        self.wb_stage.run()
        # --------------------- MEM stage --------------------
        self.mem_stage.run()
        # --------------------- EX stage ---------------------
        self.ex_stage.run()
        # --------------------- ID stage ---------------------
        self.id_stage.run()
        # --------------------- IF stage ---------------------
        self.if_stage.run()

        self.myRF.outputRF_FS(self.cycle)        # dump RF
        self.printState(self.state, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...
        self.inst_FS += int(current_instr != self.state.ID.instr)
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.__dict__().items()])
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.__dict__().items()])
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.__dict__().items()])
        printstate.extend(["ME." + key + ": " + str(val) + "\n" for key, val in state.MEM.__dict__().items()])
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.__dict__().items()])

        if cycle == 0:
            perm = "w"
        else:
            perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


def process_testcases(input_dir, output_base_dir):
    # Auto-discover and sort all testcase folders
    testcase_dirs = [d for d in os.listdir(input_dir) if d.startswith("testcase")]
    testcase_dirs.sort(key=lambda x: int(x.replace("testcase", "")))  # testcase+number

    for testcase in testcase_dirs:
        testcase_input_dir = os.path.join(input_dir, testcase)
        testcase_output_dir = os.path.join(output_base_dir, testcase)

        # If the output directory does not exist, create it
        if not os.path.exists(testcase_output_dir):
            os.makedirs(testcase_output_dir)

        # Run handler functions (e.g., run the simulator)
        run_simulation(testcase_input_dir)

        # Move the output files to the newly created folder, excluding dmem.txt and imem.txt
        for file in os.listdir(testcase_input_dir):
            if file not in ["dmem.txt", "imem.txt"]:
                shutil.move(os.path.join(testcase_input_dir, file), testcase_output_dir)


def clear_directory(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Delete everything in the directory
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Deleting files
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    print("Previous outputs have been cleared")


def run_simulation(ioDir):
    # common imem
    imem = InsMem("Imem", ioDir)

    # single stage processor
    dmem_ss = DataMem("SS", ioDir)

    ssCore = SingleStageCore(ioDir, imem, dmem_ss)

    while True:
        if not ssCore.halted:
            ssCore.step()

        if ssCore.halted:
            ssCore.myRF.outputRF(ssCore.cycle)  # output file of registers after last cycle
            ssCore.printState(ssCore.nextState, ssCore.cycle)  # print states after last cycle
            ssCore.cycle += 1
            break

    # dump SS data mem.
    dmem_ss.outputDataMem()

    # print performance metrics in file
    dmem_fs = DataMem("FS", ioDir)

    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while (True):
        if not fsCore.halted:
            fsCore.step()

        if fsCore.halted:
            break

    # dump FS data mem.
    dmem_fs.outputDataMem()

    Performance_metrics(ioDir, ssCore, fsCore)
    pass


def Performance_metrics(opFilePath: str, ss: SingleStageCore, fs: FiveStageCore):
    ss_metrics = [
        "Performance of Single Stage:",
        f"#Cycles -> {ss.cycle}",
        f"#Instructions -> {ss.inst}",
        f"CPI -> {ss.cycle / ss.inst}",
        f"IPC -> {ss.inst / ss.cycle}",
    ]

    fs_metrics = [
        "Performance of Five Stage:",
        f"#Cycles -> {fs.cycle}",
        f"#Instructions -> {fs.inst_FS + 1}",
        f"CPI -> {fs.cycle / (fs.inst_FS + 1)}",
        f"IPC -> {(fs.inst_FS + 1) / fs.cycle}",
    ]

    with open(opFilePath + os.sep + "PerformanceMetrics_Result.txt", "w") as f:
        f.write("\n".join(ss_metrics) + "\n" + "\n" + "\n".join(fs_metrics))


if __name__ == "__main__":

    # parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    # the current directory for input
    ioDir = os.path.abspath(args.iodir)
    parentDir = os.path.dirname(ioDir)

    outputDir = os.path.join(parentDir, 'output_zg2609')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    inputDir = os.path.join(parentDir, 'input')

    process_testcases(inputDir, outputDir)
