from npu.build.wslbuilder import WSLBuilder
from npu.build.kernel import Kernel
from npu.build.utils import wsl_prefix
from npu.build import wslpath
from typing import Dict,List
import subprocess
import re

def wslcall(cmd:str)->str:
    cmdlist = cmd.split(" ")
    try:
        output = subprocess.check_output(cmdlist, stderr=subprocess.STDOUT)
        return output.decode()
    except subprocess.CalledProcessError as e:
        print(f"ERROR! WSL failed \n\n{e.output.decode()}") 
        raise e

def wslcall_cmdlist(cmdlist:List[str])->str:
    try:
        output = subprocess.check_output(cmdlist, stderr=subprocess.STDOUT)
        return output.decode()
    except subprocess.CalledProcessError as e:
        print(f"ERROR! WSL failed \n\n{e.output.decode()}") 
        raise e

class CreateIpuLogDecoder(WSLBuilder):

    def __init__(self, app):
        """ Build a printf decoder for a kernel object """
        super().__init__()

        self._app = app
        self._ofile_win = self._app.ab.build_path
        self._ofile = f"{wslpath(self._ofile_win)}/core_0_2.elf"
        self._strings_command = f"{wsl_prefix()}strings --radix=x -a {self._ofile}"
        self._object_strings_str = wslcall(self._strings_command)
        self.rooffset = int(self._get_rooffset(),16)
        self.strings = self._gen_string_dict(self._object_strings_str,self.rooffset)
        self.objdump = self._get_objdump()
    
    def decode(self, bo)->List[str]:
        """ Decode the output buffer into messages """
        logs = []
        for i in range(len(bo)):
            if bo[i] in self.strings:
                num_format_specifiers = len([m.start() for m in re.finditer('%(?![%])', self.strings[bo[i]])])
                parameters=bo[i+1:(i+1)+num_format_specifiers]
                try:
                    logs.append(self.strings[bo[i]] % tuple(parameters))
                except:
                    pass
        return logs
    
    def _get_rooffset(self)->str:
        #s = [f"{wsl_prefix()}readelf", "-S", f"{self._ofile}", "|", "grep", ".rodata"]
        s = [f"{wsl_prefix()}readelf", "-S", f"{self._ofile}"]
        out = wslcall_cmdlist(s)
        pattern = r"\s*\[\s*[0-9]+\]\s*\.rodata\.DMb\.1\s*PROGBITS\s*([0-9a-z]+)"
        match = re.search(pattern, out)
        if match:
            return match.group(1)
        return "70A00"

    def _get_objdump(self)->str:
        s = f"{wsl_prefix()}objdump -s -j .rodata.DMb.1 {self._ofile}"
        out = wslcall(s)
        return out
    
    def _gen_string_dict(self, str_string:str, offset:int=0)->Dict[int,str]:
        lines = self._object_strings_str.split('\n')
        result = {}
        first = True
        first_val=0
        for line in lines:
            l = line.lstrip()
            try:
                hex_num, text = l.split(' ', 1)
                if first:
                    first_val = int(hex_num,16)
                    result[offset] = text
                    first = False
                    #result[int(hex_num,16)] = text
                else:
                    result[(int(hex_num,16) - first_val)+offset] = text
                    #result[int(hex_num,16)] = text
            except:
                    pass
        return result
