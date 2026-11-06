"""Fingerprint runtime ELF sections without debug source-directory differences."""
import hashlib
from pathlib import Path
import struct

def sections(path):
    data=Path(path).read_bytes()
    if data[:6]!=b'\x7fELF\x02\x01':raise ValueError('Expected little-endian ELF64')
    offset=struct.unpack_from('<Q',data,40)[0]
    size,count,names_index=struct.unpack_from('<HHH',data,58)
    headers=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
    header=headers[names_index];names=data[header[4]:header[4]+header[5]]
    result={}
    for header in headers:
        name=names[header[0]:].split(b'\0',1)[0].decode()
        # Allocatable data/code, runtime execution metadata and relocations may affect execution.
        # Debug records and their relocations carry source paths but are not loaded as code/data.
        selected=bool(header[2]&2) or name in ['.ze_info','.note.intelgt.compat'] or name.startswith(('.rel.text','.rela.text','.rel.data','.rela.data'))
        if selected:result[name]=(header[1],header[2],header[3],data[header[4]:header[4]+header[5]])
    if not any(name.startswith('.text.') for name in result):raise ValueError('No native kernel text')
    return result

def fingerprint(path):
    digest=hashlib.sha256()
    for name,(kind,flags,address,data) in sorted(sections(path).items()):
        digest.update(name.encode()+b'\0'+struct.pack('<QQQQ',kind,flags,address,len(data))+data)
    return digest.hexdigest()
