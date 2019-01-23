package pt.ul.fc.bytec;

import java.util.ArrayList;
import java.util.List;

public class MethodDescriptor {

    public static final Object[] PRIMITIVES = new Object[]{
            Byte.class,
            Character.class,
            Double.class,
            Float.class,
            Integer.class,
            Long.class,
            Short.class,
            Boolean.class,
            new ArrayDescriptor(null)
    };


    public final Object returnType;
    public final Object[] args;

    public static class ArrayDescriptor {
        public final Object innertype;

        public ArrayDescriptor(Object innertype) {
            this.innertype = innertype;
        }
    }

    public MethodDescriptor(String methodDescriptor) {

        char[] chars = methodDescriptor.toCharArray();
        List<Object> args = new ArrayList<>();

        int[] index = {1}; //skip paren
        while (chars[index[0]] != ')') {
            Object converted = decodeDescriptor(methodDescriptor, index);
            args.add(converted);
        }
        index[0]++;
        returnType = decodeDescriptor(methodDescriptor, index);
        this.args = args.toArray();
    }

    public static String stringify(Object o) {
        if (o == Byte.class) {
            return "int8_t";
        } else if (o == Character.class) {
            return "uint16_t";
        } else if (o == Double.class) {
            return "double";
        } else if (o == Float.class) {
            return "float";
        } else if (o == Integer.class) {
            return "int32_t";
        } else if (o == Long.class) {
            return "int64_t";
        } else if (o == Short.class) {
            return "int16_t";
        } else if (o == Boolean.class) {
            return "bool";
        } else if (o == Void.class) {
            return "void";
        } else if (o instanceof ArrayDescriptor) {
            return "void*";
        } else {
            System.err.println("idk lmao " + o + " " + o.getClass());
            return null;
        }
    }

    public static String getSafeName(Object o) {
        if (o == Byte.class) {
            return "B";
        } else if (o == Character.class) {
            return "C";
        } else if (o == Double.class) {
            return "D";
        } else if (o == Float.class) {
            return "F";
        } else if (o == Integer.class) {
            return "I";
        } else if (o == Long.class) {
            return "J";
        } else if (o == Short.class) {
            return "S";
        } else if (o == Boolean.class) {
            return "Z";
        } else if (o == Void.class) {
            return "V";
        } else {
            return "L";
        }
    }

    public static Object decodeDescriptor(String s, int[] index) {
        char c = s.charAt(index[0]);
        index[0]++;
        {
            Object primitiveDescriptor = decodePrimitiveDescriptor(c);
            if (primitiveDescriptor != null) {
                return primitiveDescriptor;
            }
        }
        {
            if (c == '[') {
                return new ArrayDescriptor(decodeDescriptor(s, index));
            }
        }
        throw new RuntimeException("Could not decode descriptor: " + s.substring(index[0] - 1));
    }

    private static Object decodePrimitiveDescriptor(char c) {
        Object converted = null;
        switch (c) {
            case 'B': {
                converted = Byte.class;
                break;
            }
            case 'C': {
                converted = Character.class;
                break;
            }
            case 'D': {
                converted = Double.class;
                break;
            }
            case 'F': {
                converted = Float.class;
                break;
            }
            case 'I': {
                converted = Integer.class;
                break;
            }
            case 'J': {
                converted = Long.class;
                break;
            }
            case 'S': {
                converted = Short.class;
                break;
            }
            case 'Z': {
                converted = Boolean.class;
                break;
            }
            case 'V': {
                converted = Void.class;
                break;
            }
        }
        return converted;
    }
}
