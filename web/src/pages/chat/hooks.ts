import { ChatSearchParams, MessageType } from '@/constants/chat';
import { fileIconMap } from '@/constants/common';
import {
  useFetchManualConversation,
  useFetchManualDialog,
  useFetchNextConversation,
  useFetchNextConversationList,
  useFetchNextDialog,
  useGetChatSearchParams,
  useRemoveNextConversation,
  useRemoveNextDialog,
  useSetNextDialog,
  useUpdateNextConversation,
} from '@/hooks/chat-hooks';
import {
  useSetModalState,
  useShowDeleteConfirm,
  useTranslate,
} from '@/hooks/common-hooks';
import {
  useRegenerateMessage,
  useSelectDerivedMessages,
  useSendMessageWithSse,
} from '@/hooks/logic-hooks';
import { IConversation, IDialog, Message } from '@/interfaces/database/chat';
import { getFileExtension } from '@/utils';
import api from '@/utils/api';
import { getConversationId } from '@/utils/chat';
import { useMutationState } from '@tanstack/react-query';
import { get } from 'lodash';
import trim from 'lodash/trim';
import {
  ChangeEventHandler,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { useSearchParams } from 'umi';
import { v4 as uuid } from 'uuid';
import {
  IClientConversation,
  IMessage,
  VariableTableDataType,
} from './interface';

export const useSetChatRouteParams = () => {
  const [currentQueryParameters, setSearchParams] = useSearchParams();
  const newQueryParameters: URLSearchParams = useMemo(
    () => new URLSearchParams(currentQueryParameters.toString()),
    [currentQueryParameters],
  );

  const setConversationIsNew = useCallback(
    (value: string) => {
      newQueryParameters.set(ChatSearchParams.isNew, value);
      setSearchParams(newQueryParameters);
    },
    [newQueryParameters, setSearchParams],
  );

  const getConversationIsNew = useCallback(() => {
    return newQueryParameters.get(ChatSearchParams.isNew);
  }, [newQueryParameters]);

  return { setConversationIsNew, getConversationIsNew };
};

export const useSetNewConversationRouteParams = () => {
  const [currentQueryParameters, setSearchParams] = useSearchParams();
  const newQueryParameters: URLSearchParams = useMemo(
    () => new URLSearchParams(currentQueryParameters.toString()),
    [currentQueryParameters],
  );

  const setNewConversationRouteParams = useCallback(
    (conversationId: string, isNew: string) => {
      newQueryParameters.set(ChatSearchParams.ConversationId, conversationId);
      newQueryParameters.set(ChatSearchParams.isNew, isNew);
      setSearchParams(newQueryParameters);
    },
    [newQueryParameters, setSearchParams],
  );

  return { setNewConversationRouteParams };
};

export const useSelectCurrentDialog = () => {
  const data = useMutationState({
    filters: { mutationKey: ['fetchDialog'] },
    select: (mutation) => {
      return get(mutation, 'state.data.data', {});
    },
  });

  return (data.at(-1) ?? {}) as IDialog;
};

export const useSelectPromptConfigParameters = (): VariableTableDataType[] => {
  const { data: currentDialog } = useFetchNextDialog();

  const finalParameters: VariableTableDataType[] = useMemo(() => {
    const parameters = currentDialog?.prompt_config?.parameters ?? [];
    if (!currentDialog.id) {
      // The newly created chat has a default parameter
      return [{ key: uuid(), variable: 'knowledge', optional: false }];
    }
    return parameters.map((x) => ({
      key: uuid(),
      variable: x.key,
      optional: x.optional,
    }));
  }, [currentDialog]);

  return finalParameters;
};

export const useDeleteDialog = () => {
  const showDeleteConfirm = useShowDeleteConfirm();

  const { removeDialog } = useRemoveNextDialog();

  const onRemoveDialog = (dialogIds: Array<string>) => {
    showDeleteConfirm({ onOk: () => removeDialog(dialogIds) });
  };

  return { onRemoveDialog };
};

export const useHandleItemHover = () => {
  const [activated, setActivated] = useState<string>('');

  const handleItemEnter = (id: string) => {
    setActivated(id);
  };

  const handleItemLeave = () => {
    setActivated('');
  };

  return {
    activated,
    handleItemEnter,
    handleItemLeave,
  };
};

export const useEditDialog = () => {
  const [dialog, setDialog] = useState<IDialog>({} as IDialog);
  const { fetchDialog } = useFetchManualDialog();
  const { setDialog: submitDialog, loading } = useSetNextDialog();

  const {
    visible: dialogEditVisible,
    hideModal: hideDialogEditModal,
    showModal: showDialogEditModal,
  } = useSetModalState();

  const hideModal = useCallback(() => {
    setDialog({} as IDialog);
    hideDialogEditModal();
  }, [hideDialogEditModal]);

  const onDialogEditOk = useCallback(
    async (dialog: IDialog) => {
      const ret = await submitDialog(dialog);

      if (ret === 0) {
        hideModal();
      }
    },
    [submitDialog, hideModal],
  );

  const handleShowDialogEditModal = useCallback(
    async (dialogId?: string) => {
      if (dialogId) {
        const ret = await fetchDialog(dialogId);
        if (ret.retcode === 0) {
          setDialog(ret.data);
        }
      }
      showDialogEditModal();
    },
    [showDialogEditModal, fetchDialog],
  );

  const clearDialog = useCallback(() => {
    setDialog({} as IDialog);
  }, []);

  return {
    dialogSettingLoading: loading,
    initialDialog: dialog,
    onDialogEditOk,
    dialogEditVisible,
    hideDialogEditModal: hideModal,
    showDialogEditModal: handleShowDialogEditModal,
    clearDialog,
  };
};

//#region conversation

export const useSelectDerivedConversationList = () => {
  const { t } = useTranslate('chat');

  const [list, setList] = useState<Array<IConversation>>([]);
  const { data: currentDialog } = useFetchNextDialog();
  const { data: conversationList, loading } = useFetchNextConversationList();
  const { dialogId } = useGetChatSearchParams();
  const prologue = currentDialog?.prompt_config?.prologue ?? '';
  const { setNewConversationRouteParams } = useSetNewConversationRouteParams();

  const addTemporaryConversation = useCallback(() => {
    const conversationId = getConversationId();
    setList((pre) => {
      if (dialogId) {
        setNewConversationRouteParams(conversationId, 'true');
        const nextList = [
          {
            id: conversationId,
            name: t('newConversation'),
            dialog_id: dialogId,
            is_new: true,
            message: [
              {
                content: prologue,
                role: MessageType.Assistant,
              },
            ],
          } as any,
          ...conversationList,
        ];
        return nextList;
      }

      return pre;
    });
  }, [conversationList, dialogId, prologue, t, setNewConversationRouteParams]);

  // When you first enter the page, select the top conversation card

  useEffect(() => {
    setList([...conversationList]);
  }, [conversationList]);

  return { list, addTemporaryConversation, loading };
};

export const useSetConversation = () => {
  const { dialogId } = useGetChatSearchParams();
  const { updateConversation } = useUpdateNextConversation();

  const setConversation = useCallback(
    async (
      message: string,
      isNew: boolean = false,
      conversationId?: string,
    ) => {
      const data = await updateConversation({
        dialog_id: dialogId,
        name: message,
        is_new: isNew,
        conversation_id: conversationId,
        message: [
          {
            role: MessageType.Assistant,
            content: message,
          },
        ],
      });

      return data;
    },
    [updateConversation, dialogId],
  );

  return { setConversation };
};

export const useSelectNextMessages = (selectedValue: string) => {
  const {
    ref,
    setDerivedMessages,
    derivedMessages,
    addNewestAnswer,
    addNewestQuestion,
    removeLatestMessage,
    removeMessageById,
    removeMessagesAfterCurrentMessage,
  } = useSelectDerivedMessages(selectedValue);
  const { data: conversation, loading } = useFetchNextConversation();
  const { data: dialog } = useFetchNextDialog();
  const { conversationId, dialogId, isNew } = useGetChatSearchParams();
  //

  const addPrologue = useCallback(() => {
    console.log(selectedValue);
    let prologue = dialog.prompt_config?.prologue;

    if (selectedValue === "CMDB") {
      prologue = `
### CMDB 数据库
| 主机名        | 运行环境 | 可用域 | 系统角色                | 虚拟化角色 | CPU | 内存    | 操作系统 | 操作系统版本         | Kernel                     | 系统架构 | IP地址                                      |
|--------------|----------|--------|-------------------------|------------|-----|---------|----------|----------------------|----------------------------|-----------|---------------------------------------------|
| baremetal02  | 生产     | tok04  | kvm                     | 物理机     | 192 | 516 GB  | Ubuntu   | 20.04                | 5.4.0-88-generic           | x86_64     | 192.168.122.110, 10.88.0.1, 128.168.65.99, 10.192.2 |
| baremetal01  | 生产     | dal10  | nvidia, machine-learning, xiaoji | 物理机 | 192 | 258 GB  | Ubuntu   | 22.04                | 5.15.0-112-generic         | x86_64     | 192.168.67.2, 172.18.0.1, 10.171.248.164, 172.17.0 |
| ansible-builder | 测试   | tok04  | elasticsearch, grafana  | 虚拟机     | 2   | 3.73 GB | CentOS   | 8.0                  | 4.18.0-358.el8.x86_64      | x86_64     | 10.88.0.1, 172.16.0.2, 192.168.69.77        |
| virtualserver01 | 准生产 | dal10  | nginx                   | 虚拟机     | 1   | 1.94 GB | Ubuntu   | 24.04                | 6.8.0-1011-ibm             | x86_64     | 10.171.248.150, 169.61.232.18, 192.168.67.1  |
| prom         | 测试     | tok04  | prometheus, grafana     | 虚拟机     | 4   | 15.7 GB | RedHat   | 9.4                  | 5.14.0-427.13.1.el9_4.x86_64 | x86_64  | 10.88.0.1, 192.168.70.100                    |
| pg1          | 测试     | tok04  | postgresql, ha_cluster  | 虚拟机     | 4   | 15.7 GB | RedHat   | 9.4                  | 5.14.0-427.13.1.el9_4.x86_64 | x86_64  | 192.168.70.211, 192.168.70.210               |
`;
    }

    const nextMessage = {
      role: MessageType.Assistant,
      content: prologue,
      id: uuid(),
      selectedSkill: selectedValue,
    } as IMessage;
    setDerivedMessages([nextMessage]);
  }, [isNew, dialog, dialogId, setDerivedMessages]);

  useEffect(() => {
    addPrologue();
  }, [addPrologue]);

  useEffect(() => {
    if (
      conversationId &&
      isNew !== 'true' &&
      conversation.message?.length > 0
    ) {
      console.log(conversation.message)
      setDerivedMessages(conversation.message);
    }
    if (!conversationId) {

      setDerivedMessages([]);
    }
  }, [conversation.message, conversationId, setDerivedMessages, isNew]);

  return {
    ref,
    derivedMessages,
    loading,
    addNewestAnswer,
    addNewestQuestion,
    removeLatestMessage,
    removeMessageById,
    removeMessagesAfterCurrentMessage,
  };
};

export const useHandleMessageInputChange = () => {
  const [value, setValue] = useState('');

  const handleInputChange: ChangeEventHandler<HTMLInputElement> = (e) => {
    const value = e.target.value;
    const nextValue = value.replaceAll('\\n', '\n').replaceAll('\\t', '\t');
    setValue(nextValue);
  };

  return {
    handleInputChange,
    value,
    setValue,
  };
};

export const useSendNextMessage = (
  controller: AbortController,
  selectedValue: string,
) => {
  const { setConversation } = useSetConversation();
  const { conversationId, isNew } = useGetChatSearchParams();
  const { handleInputChange, value, setValue } = useHandleMessageInputChange();

  const { send, answer, done } = useSendMessageWithSse(
    api.completeConversation,
  );
  const {
    ref,
    derivedMessages,
    loading,
    addNewestAnswer,
    addNewestQuestion,
    removeLatestMessage,
    removeMessageById,
    removeMessagesAfterCurrentMessage,
  } = useSelectNextMessages(selectedValue);
  const { setConversationIsNew, getConversationIsNew } =
    useSetChatRouteParams();

  const sendMessage = useCallback(
    async ({
      message,
      currentConversationId,
      messages,
      selectedSkill,
    }: {
      message: Message;
      currentConversationId?: string;
      messages?: Message[];
      selectedSkill: string;
    }) => {
      const res = await send(
        {
          prompt: message.content,
          selectedSkill: selectedSkill,
          conversation_id: currentConversationId ?? conversationId,
          messages: [...(messages ?? derivedMessages ?? []), message],
        },

        controller,
      );

      if (res && (res?.response.status !== 200 || res?.data?.retcode !== 0)) {
        // cancel loading
        setValue(message.content);
        console.info('removeLatestMessage111');
        removeLatestMessage();
      }
    },
    [
      derivedMessages,
      conversationId,
      removeLatestMessage,
      setValue,
      send,
      controller,
    ],
  );

  const handleSendMessage = useCallback(
    async (message: Message) => {
      const isNew = getConversationIsNew();

      if (isNew !== 'true') {
        sendMessage({ message, selectedSkill: message.selectedSkill });
      } else {
        const data = await setConversation(
          message.content,
          true,
          conversationId,
        );
        if (data.retcode === 0) {
          setConversationIsNew('');
          const id = data.data.id;
          // currentConversationIdRef.current = id;
          sendMessage({
            message,
            currentConversationId: id,
            messages: data.data.message,
            selectedSkill: message.selectedSkill,
          });
        }
      }
    },
    [
      setConversation,
      sendMessage,
      setConversationIsNew,
      getConversationIsNew,
      conversationId,
    ],
  );

  const { regenerateMessage } = useRegenerateMessage({
    removeMessagesAfterCurrentMessage,
    sendMessage,
    messages: derivedMessages,
  });

  useEffect(() => {
    //  #1289
    if (answer.answer && conversationId && isNew !== 'true') {
      answer.selectedSkill = selectedValue;
      addNewestAnswer(answer);
    }
  }, [answer, addNewestAnswer, conversationId, isNew]);

  const handlePressEnter = useCallback(
    (documentIds: string[]) => {
      if (trim(value) === '') return;
      const id = uuid();
      console.log('0----' + selectedValue);
      console.log(documentIds);
      console.log('1----press enter ');
      if (selectedValue === "LOG" || selectedValue == "日志分析") {
        if (documentIds.length <= 0) {
          return
        }
      }

      addNewestQuestion({
        content: value,
        doc_ids: documentIds,
        id,
        role: MessageType.User,
        selectedSkill: selectedValue,
      });
      if (done) {
        setValue('');
        handleSendMessage({
          id,
          content: value.trim(),
          role: MessageType.User,
          doc_ids: documentIds,
          selectedSkill: selectedValue,
        });
      }
    },
    [addNewestQuestion, handleSendMessage, done, setValue, value],
  );

  return {
    handlePressEnter,
    handleInputChange,
    value,
    setValue,
    regenerateMessage,
    sendLoading: !done,
    loading,
    ref,
    derivedMessages,
    removeMessageById,
  };
};

export const useGetFileIcon = () => {
  const getFileIcon = (filename: string) => {
    const ext: string = getFileExtension(filename);
    const iconPath = fileIconMap[ext as keyof typeof fileIconMap];
    return `@/assets/svg/file-icon/${iconPath}`;
  };

  return getFileIcon;
};

export const useDeleteConversation = () => {
  const showDeleteConfirm = useShowDeleteConfirm();
  const { removeConversation } = useRemoveNextConversation();

  const deleteConversation = (conversationIds: Array<string>) => async () => {
    const ret = await removeConversation(conversationIds);

    return ret;
  };

  const onRemoveConversation = (conversationIds: Array<string>) => {
    showDeleteConfirm({ onOk: deleteConversation(conversationIds) });
  };

  return { onRemoveConversation };
};

export const useRenameConversation = () => {
  const [conversation, setConversation] = useState<IClientConversation>(
    {} as IClientConversation,
  );
  const { fetchConversation } = useFetchManualConversation();
  const {
    visible: conversationRenameVisible,
    hideModal: hideConversationRenameModal,
    showModal: showConversationRenameModal,
  } = useSetModalState();
  const { updateConversation, loading } = useUpdateNextConversation();

  const onConversationRenameOk = useCallback(
    async (name: string) => {
      const ret = await updateConversation({
        ...conversation,
        conversation_id: conversation.id,
        name,
        is_new: false,
      });

      if (ret.retcode === 0) {
        hideConversationRenameModal();
      }
    },
    [updateConversation, conversation, hideConversationRenameModal],
  );

  const handleShowConversationRenameModal = useCallback(
    async (conversationId: string) => {
      const ret = await fetchConversation(conversationId);
      if (ret.retcode === 0) {
        setConversation(ret.data);
      }
      showConversationRenameModal();
    },
    [showConversationRenameModal, fetchConversation],
  );

  return {
    conversationRenameLoading: loading,
    initialConversationName: conversation.name,
    onConversationRenameOk,
    conversationRenameVisible,
    hideConversationRenameModal,
    showConversationRenameModal: handleShowConversationRenameModal,
  };
};

export const useGetSendButtonDisabled = () => {
  const { dialogId, conversationId } = useGetChatSearchParams();

  return dialogId === '' || conversationId === '';
};

export const useSendButtonDisabled = (value: string) => {
  return trim(value) === '';
};

export const useCreateConversationBeforeUploadDocument = () => {
  const { setConversation } = useSetConversation();
  const { setConversationIsNew } = useSetChatRouteParams();
  const { dialogId, conversationId, isNew } = useGetChatSearchParams();

  const createConversationBeforeUploadDocument = useCallback(
    async (message: string) => {
      const data = await setConversation(message, true, conversationId);
      if (data.retcode === 0) {
        setConversationIsNew('');
      }
      return data;
    },
    [setConversation, conversationId, setConversationIsNew],
  );

  return {
    dialogId,
    createConversationBeforeUploadDocument:
      isNew === 'true' ? createConversationBeforeUploadDocument : undefined,
  };
};

//#endregion
