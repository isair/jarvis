#!/usr/bin/env python3
# Generates .pbzero.h/.cc and .gen.h/.cc for the vendored perfetto protos,
# mirroring src/protozero/protoc_plugin/{protozero,cppgen}_plugin.cc.
# Usage: gen_perf_protos.py DS_FILE... PERF_ROOT
import os
import sys

from google.protobuf import descriptor_pb2

L_OPT, L_REP = 1, 3
T_DOUBLE, T_FLOAT, T_INT64, T_UINT64, T_INT32, T_FIXED64, T_FIXED32, T_BOOL = 1, 2, 3, 4, 5, 6, 7, 8
T_STRING, T_MESSAGE, T_BYTES, T_UINT32, T_ENUM, T_SFIXED32, T_SFIXED64, T_SINT32, T_SINT64 = 9, 11, 12, 13, 14, 15, 16, 17, 18

_DEC = {T_BOOL: ("as_bool", "bool"),
        T_SFIXED32: ("as_int32", "int32_t"), T_INT32: ("as_int32", "int32_t"),
        T_SINT32: ("as_sint32", "int32_t"),
        T_SFIXED64: ("as_int64", "int64_t"), T_INT64: ("as_int64", "int64_t"),
        T_SINT64: ("as_sint64", "int64_t"),
        T_FIXED32: ("as_uint32", "uint32_t"), T_UINT32: ("as_uint32", "uint32_t"),
        T_FIXED64: ("as_uint64", "uint64_t"), T_UINT64: ("as_uint64", "uint64_t"),
        T_FLOAT: ("as_float", "float"), T_DOUBLE: ("as_double", "double"),
        T_ENUM: ("as_int32", "int32_t"),
        T_STRING: ("as_string", "::protozero::ConstChars"),
        T_MESSAGE: ("as_bytes", "::protozero::ConstBytes"),
        T_BYTES: ("as_bytes", "::protozero::ConstBytes")}
_WIRE = {T_INT64: "kVarInt", T_UINT64: "kVarInt", T_INT32: "kVarInt",
         T_BOOL: "kVarInt", T_UINT32: "kVarInt", T_ENUM: "kVarInt",
         T_SINT32: "kVarInt", T_SINT64: "kVarInt", T_FIXED32: "kFixed32",
         T_SFIXED32: "kFixed32", T_FLOAT: "kFixed32", T_FIXED64: "kFixed64",
         T_SFIXED64: "kFixed64", T_DOUBLE: "kFixed64",
         T_STRING: "kLengthDelimited", T_BYTES: "kLengthDelimited",
         T_MESSAGE: "kLengthDelimited"}
_PBFF = {T_INT64: "PackedVarInt", T_UINT64: "PackedVarInt", T_INT32: "PackedVarInt",
         T_BOOL: "PackedVarInt", T_UINT32: "kVarIntBuf", T_UINT64: "PackedVarInt",
         T_ENUM: "PackedVarInt",
         T_SINT32: "PackedVarInt", T_SINT64: "PackedVarInt",
         T_FIXED32: "PackedFixedSizeInt<uint32_t>", T_SFIXED32: "PackedFixedSizeInt<int32_t>",
         T_FLOAT: "PackedFixedSizeInt<float>", T_FIXED64: "PackedFixedSizeInt<uint64_t>",
         T_SFIXED64: "PackedFixedSizeInt<int64_t>",
         T_DOUBLE: "PackedFixedSizeInt<double>"}
_PBFF[T_UINT32] = "PackedVarInt"
_SCHEMA = {T_BOOL: "kBool", T_INT32: "kInt32", T_INT64: "kInt64", T_UINT32: "kUint32",
           T_UINT64: "kUint64", T_SINT32: "kSint32", T_SINT64: "kSint64",
           T_FIXED32: "kFixed32", T_FIXED64: "kFixed64", T_SFIXED32: "kSfixed32",
           T_SFIXED64: "kSfixed64", T_FLOAT: "kFloat", T_DOUBLE: "kDouble",
           T_ENUM: "kEnum", T_STRING: "kString", T_MESSAGE: "kMessage", T_BYTES: "kBytes"}
_CPG = {T_INT32: "int32_t", T_SFIXED32: "int32_t", T_SINT32: "int32_t",
        T_UINT32: "uint32_t", T_FIXED32: "uint32_t", T_INT64: "int64_t",
        T_SFIXED64: "int64_t", T_SINT64: "int64_t", T_UINT64: "uint64_t",
        T_FIXED64: "uint64_t", T_BOOL: "bool", T_DOUBLE: "double", T_FLOAT: "float"}
_SET = {T_BOOL: "::protozero::internal::gen_helpers::SerializeTinyVarInt",
        T_INT32: "::protozero::internal::gen_helpers::SerializeVarInt",
        T_INT64: "::protozero::internal::gen_helpers::SerializeVarInt",
        T_UINT32: "::protozero::internal::gen_helpers::SerializeVarInt",
        T_UINT64: "::protozero::internal::gen_helpers::SerializeVarInt",
        T_ENUM: "::protozero::internal::gen_helpers::SerializeVarInt",
        T_SINT32: "::protozero::internal::gen_helpers::SerializeSignedVarInt",
        T_SINT64: "::protozero::internal::gen_helpers::SerializeSignedVarInt",
        T_FIXED32: "::protozero::internal::gen_helpers::SerializeFixed",
        T_FIXED64: "::protozero::internal::gen_helpers::SerializeFixed",
        T_SFIXED32: "::protozero::internal::gen_helpers::SerializeFixed",
        T_SFIXED64: "::protozero::internal::gen_helpers::SerializeFixed",
        T_FLOAT: "::protozero::internal::gen_helpers::SerializeFixed",
        T_DOUBLE: "::protozero::internal::gen_helpers::SerializeFixed",
        T_STRING: "::protozero::internal::gen_helpers::SerializeString",
        T_BYTES: "::protozero::internal::gen_helpers::SerializeString"}
KMAX = 999


def _camel(n):
    out, up = [], True
    for c in n:
        if c == "_":
            up = True
        elif up:
            out.append(c.upper())
            up = False
        else:
            out.append(c)
    return "".join(out)


def _cap1(s):
    return (s[0].upper() + s[1:]) if s else s


def _stripchars(s, chars):
    return "".join("_" if c in chars else c for c in s)


def _lit(n):
    return "-2147483647 - 1" if n == -2147483648 else str(n)


def _rep(f):
    return f.label == L_REP


def _packed(f):
    # Follow upstream FieldDescriptor::is_packed(): only the explicit
    # [packed=...] option counts for this plugin.
    if f.label != L_REP:
        return False
    return bool(f.options.HasField("packed") and f.options.packed)


def _cn(f):
    return _cap1(_camel(f.json_name or f.name))


def _meta_t(f):
    return "FieldMetadata_" + _cn(f)


def _meta_v(f):
    return "k" + _cn(f)


def _fnc(f):
    return "k" + _cn(f) + "FieldNumber"


def _strip(name):
    return name[:-6] if name.endswith(".proto") else name


def _which_min(e):
    mn, name = None, ""
    for v in e.value:
        if mn is None or v.number < mn:
            mn, name = v.number, v.name
    return name


def _which_max(e):
    mx, name = None, ""
    for v in e.value:
        if mx is None or v.number > mx:
            mx, name = v.number, v.name
    return name


class Gen:
    def __init__(self, files):
        self.files = files          # name -> FileDescriptorProto
        self.msg = {}               # full or tail -> (fd, cls_joined, dotted)
        self.enum = {}              # full or tail -> (fd, parent_cls_or_empty, enum)

    # -- registration (string keyed, stable across passes) ------------------
    def register_all(self):
        for fd in self.files.values():
            pre = fd.package + "." if fd.package else ""
            for m in fd.message_type:
                self._reg_msg(fd, pre + m.name, "", m)
            for e in fd.enum_type:
                self._add_enum(fd, pre + e.name, "", e)

    def _reg_msg(self, fd, dotted, cls_prefix, m):
        cls = cls_prefix + m.name
        pkg = fd.package + "." if fd.package else ""
        tail = dotted[len(pkg):] if dotted.startswith(pkg) else dotted
        for k in (dotted, tail.replace(".", "_")):
            self.msg[k] = (fd, cls, dotted)
        for n in m.nested_type:
            self._reg_msg(fd, dotted + "." + n.name, cls + "_", n)
        for e in m.enum_type:
            self._add_enum(fd, dotted + "." + e.name, cls, e)

    def _add_enum(self, fd, dotted, parent_cls, e):
        pkg = fd.package + "." if fd.package else ""
        tail = dotted[len(pkg):] if dotted.startswith(pkg) else dotted
        for k in (dotted, tail.replace(".", "_")):
            self.enum[k] = (fd, parent_cls, e)

    # -- naming -------------------------------------------------------------
    @staticmethod
    def _join(tail):
        return tail.replace(".", "_")

    @staticmethod
    def _tail(dotted, pkg):
        pre = pkg + "." if pkg else ""
        return dotted[len(pre):] if dotted.startswith(pre) else dotted

    def _cls(self, type_name):
        o = self.msg.get(type_name)
        return o[1] if o else type_name

    def _enum_name(self, type_name):
        o = self.enum.get(type_name)
        if not o:
            return type_name
        return (o[1] + "_" + o[2].name) if o[1] else o[2].name

    def _full(self, name, pfd, wrapper):
        ns = pfd.package + ("." + wrapper if wrapper else "")
        return "::" + "::".join([x for x in ns.split(".") if x]) + "::" + name

    def _cpp_name(self, tn, own_fd, wrapper):
        """Joined class/enum tail + ns prefix when cross package."""
        is_msg = tn in self.msg
        o = self.msg.get(tn) or self.enum.get(tn)
        if not o:
            return "int"
        ofd = o[0]
        if is_msg:
            j = self._join(self._tail(o[2], ofd.package))
        else:
            j = self._enum_name(tn)
        if ofd.package == own_fd.package:
            return j
        return self._full(j, ofd, wrapper)

    def _fld_t(self, f, own_fd, wrapper):
        if f.type in (T_STRING, T_BYTES):
            return "std::string"
        if f.type in _CPG:
            return _CPG[f.type]
        return self._cpp_name(f.type_name.lstrip("."), own_fd, wrapper)

    # -- one-pass DFS over a file; items carry their dotted path ------------
    def _own(self, fd):
        order = []
        pre = fd.package + "." if fd.package else ""
        stack = [[pre + m.name, m] for m in reversed(list(fd.message_type))]
        while stack:
            dotted, m = stack.pop()
            order.append((dotted, m))
            stack.extend([[dotted + "." + n.name, n] for n in m.nested_type])
        return order

    def _wraps(self, fd):
        # top-level messages that only carry "extend" blocks
        by_sc = {}
        for i in range(len(fd.extension)):
            pass
        return set()

    # ======================================================================
    # .pbzero.h
    # ======================================================================
    def _ext_of(self, fd, top_name):
        out = []
        for e in fd.extension:
            if e.extendee:
                pass
        return out

    def pb_h(self, fd):
        W = "pbzero"
        pkg = fd.package
        pre = pkg + "." if pkg else ""
        g = _stripchars((pkg + "_" + fd.name + "_H_").upper(), "./\\")
        nfull = [x for x in (pkg + "." + W).split(".") if x]
        pfx = "::" + "".join(n + "::" for n in nfull)
        L = []
        A = L.append
        A("// Autogenerated by the ProtoZero compiler plugin. DO NOT EDIT.\n")
        A("#ifndef {}\n#define {}\n\n".format(g, g))
        A("#include <stddef.h>\n#include <stdint.h>\n\n")
        A('#include "perfetto/protozero/field_writer.h"\n'
          '#include "perfetto/protozero/message.h"\n'
          '#include "perfetto/protozero/packed_repeated_fields.h"\n'
          '#include "perfetto/protozero/proto_decoder.h"\n'
          '#include "perfetto/protozero/proto_utils.h"\n')
        for di in fd.public_dependency:
            A('#include "{}.pbzero.h"\n'.format(_strip(fd.dependency[di])))
        A("\n")

        wm, wn, ext = self._collect(fd)
        # referenced (forward-declared) sets
        refm, refe = [], []
        sm, se = set(), set()
        for dotted, m in wm:
            for n in m.nested_type:
                dn = dotted + "." + n.name
                if dn in sm:
                    continue
                sm.add(dn)
                refm.append((pkg, self._join(self._tail(dn, pkg))))
        for dotted, m in wm:
            for f in m.field:
                if f.type == T_MESSAGE:
                    tn = f.type_name.lstrip(".")
                    if tn in sm:
                        continue
                    o = self.msg.get(tn)
                    if not o or o[0].name == fd.name:
                        continue
                    sm.add(tn)
                    refm.append((o[0].package, self._join(self._tail(o[2], o[0].package))))
                elif f.type == T_ENUM:
                    tn = f.type_name.lstrip(".")
                    if tn in se:
                        continue
                    o = self.enum.get(tn)
                    if not o or o[0].name == fd.name:
                        continue
                    se.add(tn)
                    jn = (o[1] + "_" + o[2].name) if o[1] else o[2].name
                    refe.append((o[0].package, jn, o[1], o[2].name))
        groups = {}
        for p, c in refm:
            groups.setdefault(p, [[], []])[0].append(c)
        for p, jn, pp, sn in refe:
            groups.setdefault(p, [[], []])[1].append((jn, pp, sn))
        for p in sorted(groups):
            nsp = [x for x in ((p + "." if p else "") + W).split(".") if x]
            for n in nsp:
                A("namespace {} {{\n".format(n))
            for c in groups[p][0]:
                A("class {};\n".format(c))
            for jn, pp, sn in groups[p][1]:
                if pp:
                    nn = "perfetto_pbzero_enum_" + pp
                    A("namespace {} {{\n}}  // namespace {}\n".format(nn, nn))
                    A("enum {} : int32_t;\n".format(sn))
                    A("using {} = {}::{};\n".format(jn, nn, sn))
                else:
                    A("enum {} : int32_t;\n".format(jn))
            for n in reversed(nsp):
                A("}} // Namespace {}.\n".format(n))
        A("\n")

        # own block
        for n in nfull:
            A("namespace {} {{\n".format(n))
        A("\n")
        ordered = []
        for e in fd.enum_type:
            ordered.append((e, ""))
        for dotted, m in wm:
            j = self._join(self._tail(dotted, pkg))
            for e in m.enum_type:
                ordered.append((e, j))
        for e, parent in ordered:
            self._pb_enum(e, parent, A, nfull, pfx, pre)
        for dotted, m in wm:
            self._pb_msg(m, dotted, fd, pfx, ext.get(dotted), A)
        for n in reversed(nfull):
            A("} // Namespace.\n")
        A("#endif  // Include guard.\n")
        return "".join(L)

    def _collect(self, fd):
        """returns [(dotted,m) of non-wrapper msgs (DFS)],
        top-level wrapper dotted names, ext map dotted->list(fld)."""
        pre = fd.package + "." if fd.package else ""
        top = list(fd.message_type)
        n_ext = {}
        for e in fd.extension:
            n_ext.setdefault(-1, []).append(e)
        # map extensions by extendee name -> wrapper detection via nested scope
        # wrapper = message whose declared extensions belong to it; DS gives us
        # fd.extension entries in file order; wrapper msgs have field_count 0
        # AND appear in message_type with matching extendee prefix... detect by
        # name matching first extension's extendee:
        exts = list(fd.extension)
        wraps = []
        used = set()
        for i, m in enumerate(top):
            if m.field or m.nested_type or m.enum_type:
                continue
            # find the first unused extendee whose joined tail == cls(name)
            for j, e in enumerate(exts):
                if j in used or e.extendee is None:
                    continue
                if self._join(self._tail(e.extendee.lstrip("."), fd.package)) == m.name:
                    wraps.append(pre + m.name)
                    n_ext.setdefault(pre + m.name, []).append(e)
                    used.add(j)
        # leftovers by extendee dotted
        for j, e in enumerate(exts):
            if j not in used and e.extendee and e.extendee.lstrip(".") != "":
                n_ext.setdefault(-2 if False else ("X", j), []).append(e)
        order = self._own(fd)
        msgs = [(d, m) for d, m in order if d not in set(wraps)]
        extmap = {k: v for k, v in n_ext.items() if isinstance(k, str)}
        return msgs, wraps, extmap

    def _pb_enum(self, e, parent, A, nfull, pfx, pre):
        cls = (parent + "_" + e.name) if parent else e.name
        if parent:
            A("namespace perfetto_pbzero_enum_{} {{\n".format(parent))
        A("enum {} : int32_t {{\n".format(e.name if parent else cls))
        for v in e.value:
            A("  {} = {},\n".format(v.name, _lit(v.number)))
        A("};\n")
        if parent:
            A("}}  // namespace perfetto_pbzero_enum_{}\n".format(parent))
            A("using {} = perfetto_pbzero_enum_{}::{};\n\n".format(cls, parent, e.name))
        A("constexpr {} {}_MIN = {}::{};\n".format(cls, cls, cls, _which_min(e)))
        A("constexpr {} {}_MAX = {}::{};\n\n".format(cls, cls, cls, _which_max(e)))
        A("PERFETTO_PROTOZERO_CONSTEXPR14_OR_INLINE\n")
        A("const char* {}_Name({}{} value) {{\n  switch (value) {{".format(cls, pfx, cls))
        for v in e.value:
            A("\n    case {}{}::{}:\n      return \"{}\";".format(pfx, cls, v.name, v.name))
        A("\n  }\n  return \"PBZERO_UNKNOWN_ENUM_VALUE\";\n}\n\n")

    def _pb_msg(self, md, dotted, fd, pfx, exts, A):
        pkg = fd.package
        cls = self._join(self._tail(dotted, pkg))
        dec = cls + "_Decoder"
        flds = list(md.field) + (list(exts) if exts else [])
        max_id = 0
        nonp = False
        for f in flds:
            if f.number > KMAX:
                continue
            max_id = max(max_id, f.number)
            if _rep(f) and not _packed(f):
                nonp = True
        for r in md.extension_range:
            if r.end - 1 <= KMAX:
                max_id = max(max_id, r.end - 1)
        A("class {} : public ::protozero::TypedProtoDecoder</*MAX_FIELD_ID=*/{},"
          " /*HAS_NONPACKED_REPEATED_FIELDS=*/{}> {{\n public:\n".format(dec, max_id, str(nonp).lower()))
        A("  {}(const uint8_t* data, size_t len) : TypedProtoDecoder(data, len) {{}}\n".format(dec))
        A("  explicit {}(const std::string& raw) : TypedProtoDecoder("
          "reinterpret_cast<const uint8_t*>(raw.data()), raw.size()) {{}}\n".format(dec))
        A("  explicit {}(const ::protozero::ConstBytes& raw) : "
          "TypedProtoDecoder(raw.data, raw.size) {{}}\n".format(dec))
        for f in flds:
            gt, ct = _DEC[f.type]
            if f.number > max_id:
                A("  // field {} omitted because its id is too high\n".format(f.name))
                continue
            A("  bool has_{}() const {{ return at<{}>().valid(); }}\n".format(f.name, f.number))
            if _packed(f):
                wt = "::protozero::proto_utils::ProtoWireType::" + _WIRE[f.type]
                A("  ::protozero::PackedRepeatedFieldIterator<{}, {}> {}(bool* parse_error_ptr) const "
                  "{{ return GetPackedRepeated<{}, {}>({}, parse_error_ptr); }}\n".format(wt, ct, f.name, wt, ct, f.number))
            elif _rep(f):
                A("  ::protozero::RepeatedFieldIterator<{}> {}() const {{ return GetRepeated<{}>({}); }}\n"
                  .format(ct, f.name, ct, f.number))
            else:
                A("  {} {}() const {{ return at<{}>().{}(); }}\n".format(ct, f.name, f.number, gt))
        A("};\n\n")
        is_wrap = bool(exts)
        A("class {} : public {} {{\n public:\n".format(
            cls, (pfx[:-2] if False else self._full(self._join(self._tail(exts[0].extendee.lstrip("."), pkg)), fd, "pbzero")) if is_wrap else "::protozero::Message"))
        if not is_wrap:
            A("  using Decoder = {}_Decoder;\n".format(cls))
        if flds:
            A("  enum : int32_t {\n")
            for f in flds:
                A("    {} = {},\n".format(_fnc(f), f.number))
            A("  };\n")
        if not is_wrap:
            A("  static constexpr const char* GetName() {{ return \"{}\"; }}\n\n".format("." + dotted))
        for n in md.nested_type:
            A("  using {} = {};\n".format(n.name, pfx + self._join(self._tail(dotted + "." + n.name, pkg))))
        for e in md.enum_type:
            jn = pfx + cls + "_" + e.name
            A("  using {} = {};\n".format(e.name, jn))
            A("  static inline const char* {}_Name({} value) {{\n    return {}_Name(value);\n  }}\n".format(
                e.name, e.name, jn))
        for e in md.enum_type:
            for v in e.value:
                A("  static constexpr {} {} = {}::{};\n".format(e.name, v.name, e.name, v.name))
        for f in flds:
            self._pb_fld(f, cls, fd, A)
        A("};\n\n")

    def _pb_fld(self, f, msg_cls, fd, A):
        mt = _meta_t(f)
        rep = "kNotRepeated" if not _rep(f) else ("kRepeatedPacked" if _packed(f) else "kRepeatedNotPacked")
        ct = self._fld_t(f, fd, "pbzero")
        A("\nusing {} =\n  ::protozero::proto_utils::FieldMetadata<\n    {},"
          "\n    ::protozero::proto_utils::RepetitionType::{},"
          "\n    ::protozero::proto_utils::ProtoSchemaType::{},"
          "\n    {},\n    {}>;\n\n".format(mt, f.number, rep, _SCHEMA[f.type], ct, msg_cls))
        A("static constexpr {} {}{{}};\n".format(mt, _meta_v(f)))
        if f.type == T_MESSAGE:
            inner = self._fld_t(f, fd, "pbzero")
            act = "add" if _rep(f) else "set"
            A("template <typename T = {}> T* {}_{}() {{\n  return BeginNestedMessage<T>({});\n}}\n\n"
              .format(inner, act, f.name, f.number))
            if f.options.lazy:
                A("void {}_{}_raw(const std::string& raw) {{\n  return AppendBytes({}, raw.data(), raw.size());\n}}\n\n"
                  .format(act, f.name, f.number))
            return
        act = "add" if _rep(f) else "set"
        if _packed(f):
            buf = "::protozero::" + _PBFF[f.type]
            A("void {}_{}(const {}& packed_buffer) {{\n  AppendBytes({}::kFieldId, packed_buffer.data(),\n"
              "            packed_buffer.size());\n}}\n".format("set", f.name, buf, mt))
            return
        if f.type == T_STRING:
            A("void {}_{}(const char* data, size_t size) {{\n  AppendBytes({}::kFieldId, data, size);\n}}\n"
              "void {}_{}(::protozero::ConstChars chars) {{\n  AppendBytes({}::kFieldId, chars.data, chars.size);\n"
              "}}\n".format(act, f.name, mt, act, f.name, mt))
        elif f.type == T_BYTES:
            A("void {}_{}(const uint8_t* data, size_t size) {{\n  AppendBytes({}::kFieldId, data, size);\n}}\n"
              "void {}_{}(::protozero::ConstBytes bytes) {{\n  AppendBytes({}::kFieldId, bytes.data, bytes.size);\n"
              "}}\n".format(act, f.name, mt, act, f.name, mt))
        A("void {}_{}({} value) {{\n  static constexpr uint32_t field_id = {}::kFieldId;\n"
          "  // Call the appropriate protozero::Message::Append(field_id, ...)\n"
          "  // method based on the type of the field.\n"
          "  ::protozero::internal::FieldWriter<\n"
          "    ::protozero::proto_utils::ProtoSchemaType::{}>\n"
          "      ::Append(*this, field_id, value);\n}}\n".format(act, f.name, ct, mt, _SCHEMA[f.type]))

    # ======================================================================
    # .gen.h
    # ======================================================================
    def gen_h(self, fd):
        pkg = fd.package
        g = _stripchars((pkg + "_" + fd.name + "_CPP_H_").upper(), "./\\")
        L = []
        A = L.append
        A("// DO NOT EDIT. Autogenerated by Perfetto cppgen_plugin\n")
        A("#ifndef {}\n#define {}\n\n".format(g, g))
        A("#include <stdint.h>\n#include <bitset>\n#include <vector>\n#include <string>\n"
          "#include <type_traits>\n\n"
          '#include "perfetto/protozero/cpp_message_obj.h"\n'
          '#include "perfetto/protozero/copyable_ptr.h"\n'
          '#include "perfetto/base/export.h"\n\n')
        A("namespace protozero {\nclass Message;\n}  // namespace protozero\n\n")

        pre = pkg + "." if pkg else ""
        wm, wraps, ext = self._collect(fd)

        # traversal: types (joined, dotted, m, ofd), enums (joined, e, ofd)
        tlist = []
        seen = set()
        tseen = set()

        def add_type(tn):
            if tn in tseen:
                return
            tseen.add(tn)
            r = self._resolve(tn)
            if not r:
                return
            rfd, rm = r
            jj = self._join(self._tail(tn, rfd.package))
            tseen.add(tn)
            tlist.append((jj, tn, rm, rfd))

        for dotted, m in wm:
            add_type(dotted)

        elist = []
        eseen = set()

        def add_enum(tn):
            if tn in eseen:
                return
            eo = self.enum.get(tn)
            if not eo:
                return
            eseen.add(tn)
            jj = (eo[1] + "_" + eo[2].name) if eo[1] else eo[2].name
            elist.append((jj, eo[2], eo[0]))

        for e in fd.enum_type:
            add_enum(pre + e.name)
        # expand fields (messages + enums) like plugin DFS
        i = 0
        while i < len(tlist):
            jj, dotted, m, mfd = tlist[i]
            i += 1
            for e in m.enum_type:
                add_enum(dotted + "." + e.name)
            for f in m.field:
                tn = f.type_name.lstrip(".")
                if f.type == T_ENUM:
                    add_enum(tn)
                elif f.type == T_MESSAGE and not f.options.lazy:
                    o = self.msg.get(tn)
                    if o:
                        add_type(tn)

        byp = {}
        for jj, dotted, m, pfd in tlist:
            byp.setdefault(pfd.package or "", []).append("class {};".format(jj))
        for jj, e, pfd in elist:
            byp.setdefault(pfd.package or "", []).append("enum {} : int;".format(jj))
        for p in sorted(byp):
            if p == "":
                continue
            ns = [x for x in p.split(".") if x] + ["gen"]
            for n in ns:
                A("namespace {} {{\n".format(n))
            for d in byp[p]:
                A(d + "\n")
            for n in reversed(ns):
                A("}}  // namespace {}\n".format(n))
            A("\n")

        nfile = [x for x in pkg.split(".") if x] + ["gen"]
        for n in nfile:
            A("namespace {} {{\n".format(n))
        # local enums + decls: file's own (non-wrapper) DFS order like plugin
        for e in fd.enum_type:
            self._gen_enum_body(e, "", A)
        for dotted, m in wm:
            pfx = self._join(self._tail(dotted, pkg))
            for e in m.enum_type:
                self._gen_enum_body(e, pfx, A)
        wset = set(wraps)
        for dotted, m in wm:
            if dotted in wset:
                continue
            self._g_decl(m, dotted, fd, A)
        for n in reversed(nfile):
            A("}}  // namespace {}\n".format(n))
        A("\n#endif  // {}\n".format(g))
        return "".join(L)

    def _resolve(self, dotted_full):
        o = self.msg.get(dotted_full)
        if not o:
            return None
        rfd = o[0]
        rpre = rfd.package + "." if rfd.package else ""
        rest = dotted_full[len(rpre):] if rpre and dotted_full.startswith(rpre) else dotted_full
        m = None
        lst = rfd.message_type
        for part in rest.split("."):
            cand = [x for x in lst if x.name == part]
            if not cand:
                return None
            m = cand[0]
            lst = m.nested_type
        return rfd, m

    def _gen_enum_body(self, e, prefix, A):
        cls = (prefix + "_" + e.name) if prefix else e.name
        inner = bool(prefix)
        A("enum {} : int {{\n".format(cls))
        for v in e.value:
            A("  {}{} = {},\n".format(cls + "_" if inner else "", v.name, _lit(v.number)))
        A("};\n\n")

    def _g_t(self, f, fd, constref=False):
        if f.type in (T_STRING, T_BYTES):
            return "const std::string&" if constref else "std::string"
        if f.type in _CPG:
            return _CPG[f.type]
        tn = f.type_name.lstrip(".")
        is_msg = tn in self.msg
        o = self.msg.get(tn) or self.enum.get(tn)
        if not o:
            return "int32_t"
        ofd = o[0]
        if is_msg:
            nm = self._join(self._tail(o[2], ofd.package))
        else:
            nm = self._enum_name(tn)
        if ofd.package and ofd.package != fd.package:
            nm = "::" + "::".join([x for x in ofd.package.split(".") if x]) + "::gen::" + nm
        return ("const " + nm + "&") if constref else nm

    def _g_decl(self, md, dotted, fd, A):
        cls = self._join(self._tail(dotted, fd.package))
        A("\nclass PERFETTO_EXPORT_COMPONENT {} : public ::protozero::CppMessageObj {{\n"
          " public:\n".format(cls))
        for n in md.nested_type:
            A("  using {} = {};\n".format(n.name, self._join(self._tail(dotted + "." + n.name, fd.package))))
        for e in md.enum_type:
            ec = cls + "_" + e.name
            A("  using {} = {};\n".format(e.name, ec))
            for v in e.value:
                A("  static constexpr auto {} = {}_{};\n".format(v.name, ec, v.name))
            A("  static constexpr auto {}_MIN = {};\n".format(e.name, ec + "_" + _which_min(e)))
            A("  static constexpr auto {}_MAX = {};\n\n".format(e.name, ec + "_" + _which_max(e)))
        if md.field:
            A("  enum FieldNumbers {\n")
            for f in md.field:
                A("    k{}FieldNumber = {},\n".format(_cn(f), f.number))
            A("  };\n\n")
        A("  {}();\n  ~{}() override;\n  {}({}&&) noexcept;\n  {}& operator=({}&&);\n"
          "  {}(const {}&);\n  {}& operator=(const {}&);\n"
          "  bool operator==(const {}&) const;\n"
          "  bool operator!=(const {}& other) const {{ return !(*this == other); }}\n"
          "\n"
          "  bool ParseFromArray(const void*, size_t) override;\n"
          "  std::string SerializeAsString() const override;\n"
          "  std::vector<uint8_t> SerializeAsArray() const override;\n"
          "  void Serialize(::protozero::Message*) const;\n".format(*([cls] * 22)))
        for f in md.field:
            ln = f.name
            A("\n")
            if f.options.lazy:
                A("  const std::string& {}_raw() const {{ return {}_; }}\n".format(ln, ln))
                A("  void set_{}_raw(std::string raw) {{ {}_ = std::move(raw); _has_field_.set({}); }}\n"
                  .format(ln, ln, f.number))
            elif not _rep(f):
                if f.type == T_MESSAGE:
                    A("  bool has_{}() const {{ return _has_field_[{}]; }}\n".format(ln, f.number))
                    A("  {} {}() const {{ return *{}_; }}\n".format(self._g_t(f, fd, True), ln, ln))
                    A("  {}* mutable_{}() {{ _has_field_.set({}); return {}_.get(); }}\n"
                      .format(self._g_t(f, fd), ln, f.number, ln))
                else:
                    A("  bool has_{}() const {{ return _has_field_[{}]; }}\n".format(ln, f.number))
                    A("  {} {}() const {{ return {}_; }}\n".format(self._g_t(f, fd, True), ln, ln))
                    A("  void set_{}({} value) {{ {}_ = value; _has_field_.set({}); }}\n"
                      .format(ln, self._g_t(f, fd, True), ln, f.number))
                    if f.type == T_BYTES:
                        A("  void set_{}(const void* p, size_t s) {{ {}_.assign(reinterpret_cast<const char*>(p), s); _has_field_.set({}); }}\n"
                          .format(ln, ln, f.number))
            else:
                ct = self._g_t(f, fd)
                A("  const std::vector<{}>& {}() const {{ return {}_; }}\n".format(ct, ln, ln))
                A("  std::vector<{}>* mutable_{}() {{ return &{}_; }}\n".format(ct, ln, ln))
                if f.type == T_MESSAGE:
                    A("  int {}_size() const;\n  void clear_{}();\n  {}* add_{}();\n".format(ln, ln, ct, ln))
                else:
                    A("  int {}_size() const {{ return static_cast<int>({}_.size()); }}\n"
                      "  void clear_{}() {{ {}_.clear(); }}\n"
                      "  void add_{}({} value) {{ {}_.emplace_back(value); }}\n"
                      "  {}* add_{}() {{ {}_.emplace_back(); return &{}_.back(); }}\n"
                      .replace("  {}* add_0() {"[1:], "").format(ln, ln, ln, ln, ln, ct, ln, ct, ln, ln, ln))
        A("\n private:\n")
        max_id = 1
        for f in md.field:
            ln = f.name
            max_id = max(max_id, f.number)
            ctj = self._g_t(f, fd)
            if f.options.lazy:
                A("  std::string {}_;  // [lazy=true]\n".format(ln))
            elif _rep(f):
                A("  std::vector<{}> {}_;\n".format(ctj, ln))
            elif f.type == T_MESSAGE:
                A("  ::protozero::CopyablePtr<{}> {}_;\n".format(ctj, ln))
            else:
                A("  {} {}_1;\n".format(ctj, ln).replace(" _1;", "_;").replace("{}_1", "{}_")
                  .format(ln) if False else "  {} {}_;\n".format(ctj, ln))
        A("\n  // Allows to preserve unknown protobuf fields for compatibility\n"
          "  // with future versions of .proto files.\n  std::string unknown_fields_;\n"
          "\n  std::bitset<{}> _has_field_{{}};\n}};\n\n".format(max_id + 1))

    # ======================================================================
    # .gen.cc
    # ======================================================================
    def gen_cc(self, fd):
        L = []
        A = L.append
        A("#include \"perfetto/protozero/gen_field_helpers.h\"\n"
          "#include \"perfetto/protozero/message.h\"\n"
          "#include \"perfetto/protozero/packed_repeated_fields.h\"\n"
          "#include \"perfetto/protozero/proto_decoder.h\"\n"
          "#include \"perfetto/protozero/scattered_heap_buffer.h\"\n")
        A("// DO NOT EDIT. Autogenerated by Perfetto cppgen_plugin\n")
        A("#if defined(__GNUC__) || defined(__clang__)\n"
          "#pragma GCC diagnostic push\n"
          "#pragma GCC diagnostic ignored \"-Wfloat-equal\"\n"
          "#endif\n")
        pre = (fd.package + ".") if fd.package else ""
        lazy = set()
        for m in fd.message_type:
            for f in m.field:
                if f.options.lazy:
                    tn = f.type_name.lstrip(".")
                    oo = self.msg.get(tn)
                    if oo:
                        lazy.add(oo[0].name)
        inc, vst, st = [], set(), [fd.name]
        while st:
            nm = st.pop()
            if nm in vst:
                continue
            vst.add(nm)
            inc.append(nm)
            pfd = self.files.get(nm)
            if pfd:
                for d in pfd.dependency:
                    if d not in vst and d not in lazy:
                        st.append(d)
        for n in inc:
            A("#include \"{}.gen.h\"\n".format(_strip(n)))
        A("\n")
        nfile = [x for x in fd.package.split(".") if x] + ["gen"]
        for n in nfile:
            A("namespace {} {{\n".format(n))
        wm, wraps, ext = self._collect(fd)
        wset = set(wraps)
        for dotted, m in wm:
            if dotted not in wset:
                self._g_def(m, dotted, fd, A)
        for n in reversed(nfile):
            A("}}  // namespace {}\n".format(n))
        A("#if defined(__GNUC__) || defined(__clang__)\n"
          "#pragma GCC diagnostic pop\n#endif\n")
        return "".join(L)

    def _g_def(self, md, dotted, fd, A):
        cls = self._join(self._tail(dotted, fd.package))
        A("\n{}::{}() = default;\n{}::~{}() = default;\n{}::{}(const {}&) = default;\n"
          "{}& {}::operator=(const {}&) = default;\n"
          "{}::{}({}&&) noexcept = default;\n{}& {}::operator=({}&&) = default;\n\n"
          .format(*([cls] * 16)))
        A("bool {}::operator==(const {}& other) const {{\n  return "
          "::protozero::internal::gen_helpers::EqualsField(unknown_fields_, other.unknown_fields_)"
          .format(cls, cls))
        for f in md.field:
            A("\n    && ::protozero::internal::gen_helpers::EqualsField({}_, other.{}_)".format(f.name, f.name))
        A(";\n}\n\n")
        for f in md.field:
            if _rep(f) and f.type == T_MESSAGE and not f.options.lazy:
                ln = f.name
                A("int {}::{}_size() const {{ return static_cast<int>({}_.size()); }}\n".format(cls, ln, ln))
                A("void {}::clear_{}() {{ {}_.clear(); }}\n".format(cls, ln, ln))
                A("{}* {}::add_{}() {{ {}_.emplace_back(); return &{}_.back(); }}\n"
                  .format(self._g_t(f, fd), cls, ln, ln, ln))
        A("bool {}::ParseFromArray(const void* raw, size_t size) {{\n".format(cls))
        for f in md.field:
            if _rep(f):
                A("  {}_.clear();\n".format(f.name))
        A("  unknown_fields_.clear();\n  bool packed_error = false;\n\n"
          "  ::protozero::ProtoDecoder dec(raw, size);\n"
          "  for (auto field = dec.ReadField(); field.valid(); field = dec.ReadField()) {\n"
          "    if (field.id() < _has_field_.size()) {\n"
          "      _has_field_.set(field.id());\n"
          "    }\n"
          "    switch (field.id()) {\n")
        for f in md.field:
            ln = f.name
            A("      case {} /* {} */:\n".format(f.number, ln))
            if f.options.lazy:
                A("        ::protozero::internal::gen_helpers::DeserializeString(field, &{}_);\n"
                  "        break;\n".format(ln))
                continue
            if _packed(f):
                wt = "::protozero::proto_utils::ProtoWireType::" + _WIRE[f.type]
                A("        if (!::protozero::internal::gen_helpers::DeserializePackedRepeated<{}, {}>"
                  "(field, &{}_)) {{\n          packed_error = true;\n        }}\n".format(wt, self._g_t(f, fd), ln))
            elif _rep(f):
                A("        {}_.emplace_back();\n".format(ln))
                if f.type == T_MESSAGE:
                    A("        {}_.back().ParseFromArray(field.data(), field.size());\n".format(ln))
                elif f.type in (T_SINT32, T_SINT64):
                    A("        field.get_signed(&{}_.back());\n".format(ln))
                elif f.type == T_STRING:
                    A("        ::protozero::internal::gen_helpers::DeserializeString(field, &{}_.back());\n"
                      .format(ln))
                else:
                    A("        field.get(&{}_.back());\n".format(ln))
            elif f.type == T_MESSAGE:
                A("        (*{}) .ParseFromArray(field.data(), field.size());\n".format(ln + "_"))
            elif f.type == T_STRING:
                A("        ::protozero::internal::gen_helpers::DeserializeString(field, &{});\n".format(ln + "_"))
            elif f.type in (T_SINT32, T_SINT64):
                A("        field.get_signed(&{});\n".format(ln + "_"))
            else:
                A("        field.get(&{});\n".format(ln + "_"))
            A("        break;\n")
        A("      default:\n"
          "        field.SerializeAndAppendTo(&unknown_fields_);\n"
          "        break;\n    }\n  }\n"
          "  return !packed_error && !dec.bytes_left();\n}\n\n")
        A("std::string {}::SerializeAsString() const {{\n"
          "  ::protozero::internal::gen_helpers::MessageSerializer msg;\n"
          "  Serialize(msg.get());\n"
          "  return msg.SerializeAsString();\n}}\n\n"
          "std::vector<uint8_t> {}::SerializeAsArray() const {{\n"
          "  ::protozero::internal::gen_helpers::MessageSerializer msg;\n"
          "  Serialize(msg.get());\n"
          "  return msg.SerializeAsArray();\n}}\n\n".format(cls, cls))
        A("void {}::Serialize(::protozero::Message* msg) const {{\n".format(cls))
        for f in md.field:
            ln = f.name + "_"
            A("  // Field {}: {}\n".format(f.number, f.name))
            if _packed(f):
                buf = "::protozero::" + _PBFF[f.type]
                A("  {{\n    {} pack;\n    for (auto& it : {})\n      pack.Append(it);\n"
                  "    msg->AppendBytes({}, pack.data(), pack.size());\n  }}\n\n".format(buf, ln, f.number))
                continue
            if _rep(f):
                A("  for (auto& it : {}) {{\n".format(ln))
                if f.type == T_MESSAGE:
                    A("    it.Serialize(msg->BeginNestedMessage<::protozero::Message>({}));\n".format(f.number))
                else:
                    A("    {}({}, it, msg);\n".format(_SET[f.type], f.number))
                A("  }\n\n")
            else:
                A("  if (_has_field_[{}]) {{\n".format(f.number))
                if f.options.lazy:
                    A("    msg->AppendString({}, {});\n".format(f.number, ln))
                elif f.type == T_MESSAGE:
                    A("    (*{}) .Serialize(msg->BeginNestedMessage<::protozero::Message>({}));\n"
                      .format(ln, f.number))
                else:
                    A("    {}({}, {}, msg);\n".format(_SET[f.type], f.number, ln))
                A("  }\n\n")
        A("  protozero::internal::gen_helpers::SerializeUnknownFields(unknown_fields_, msg);\n}\n\n")


def main():
    if len(sys.argv) < 3:
        print("usage: gen_perf_protos.py DS... PERF_ROOT")
        return 1
    files = []
    for arg in sys.argv[1:-1]:
        fdp = descriptor_pb2.FileDescriptorSet()
        with open(arg, "rb") as fh:
            fdp.ParseFromString(fh.read())
        files.extend(fdp.file)
    protos = {p.name: p for p in files}
    root = sys.argv[-1]
    n = 0
    for p in files:
        g = Gen(protos)
        g.register_all()
        base = _strip(p.name).replace("/", os.sep)
        d = os.path.join(root, os.path.dirname(base))
        os.makedirs(d or root, exist_ok=True)
        for ext, txt in ((".pbzero.h", g.pb_h(p)), (".pbzero.cc", "// Intentionally empty (crbug.com/998165)\n"),
                         (".gen.h", g.gen_h(p)), (".gen.cc", g.gen_cc(p))):
            with open(os.path.join(root, base + ext), "w", newline="\n") as fh:
                fh.write(txt)
        n += 1
    print(n, "files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
